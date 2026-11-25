"""Claim images: loading, model input, stage payloads, the batch runner, and the CLI."""

from __future__ import annotations

import asyncio
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import facticli.core.images as images_module
from facticli.averitec_submission import _run as run_batch
from facticli.averitec_submission import build_parser as build_batch_parser
from facticli.averitec_submission import image_sources
from facticli.cli import build_parser, run_check_command, run_extract_claims_command
from facticli.config import FactCheckRuntimeConfig
from facticli.core.artifacts import RunArtifacts
from facticli.core.contracts import (
    AspectFinding,
    ClaimExtractionResult,
    EvidenceSignal,
    FactCheckReport,
    InvestigationPlan,
    VeracityVerdict,
    VerificationCheck,
)
from facticli.core.images import ImageError, load_image, load_images, public_url_for, sniff_mime_type
from facticli.llm import stage_input
from facticli.payloads import IMAGE_NOTICES, images_payload, with_images
from facticli.pipeline import ClaimExtractor, FactChecker

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def _write(directory: str, name: str, data: bytes) -> str:
    path = Path(directory) / name
    path.write_bytes(data)
    return str(path)


class ImageLoadingTests(unittest.TestCase):
    def test_formats_are_recognized_by_content_not_by_name(self):
        self.assertEqual(sniff_mime_type(JPEG), "image/jpeg")
        self.assertEqual(sniff_mime_type(PNG), "image/png")
        self.assertEqual(sniff_mime_type(b"GIF89a" + b"\x00" * 8), "image/gif")
        self.assertEqual(sniff_mime_type(b"RIFF\x00\x00\x00\x00WEBPVP8 "), "image/webp")
        self.assertIsNone(sniff_mime_type(b"<svg xmlns='http://www.w3.org/2000/svg'/>"))
        with tempfile.TemporaryDirectory() as tmp:
            image = load_image(_write(tmp, "photo.png", JPEG), image_id="image_1")  # misnamed JPEG
            self.assertEqual(image.mime_type, "image/jpeg")
            self.assertTrue(image.data_url.startswith("data:image/jpeg;base64,/9j/"))
            with self.assertRaisesRegex(ImageError, "Unsupported image format"):
                load_image(_write(tmp, "drawing.svg", b"<svg/>"), image_id="image_2")

    def test_local_images_get_a_public_url_only_under_a_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "a b.jpg", JPEG)
            first, second = load_images([path, _write(tmp, "c.png", PNG)], url_base="https://host.example/imgs/")
            self.assertEqual((first.image_id, second.image_id), ("image_1", "image_2"))
            self.assertEqual(first.public_url, "https://host.example/imgs/a%20b.jpg")
            self.assertIsNone(load_image(path, image_id="image_1").public_url)
        self.assertIsNone(public_url_for("x.jpg", None))

    def test_artifacts_keep_metadata_but_never_the_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = load_image(_write(tmp, "a.jpg", JPEG), image_id="image_1")
        dumped = image.model_dump()
        self.assertNotIn("data", dumped)
        self.assertEqual((dumped["byte_size"], len(dumped["sha256"])), (len(JPEG), 64))
        artifacts = RunArtifacts(claim="c", normalized_claim="c", images=[image])
        self.assertNotIn("base64", artifacts.model_dump_json())

    def test_unusable_images_are_refused_with_a_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ImageError, "not found"):
                load_image(str(Path(tmp) / "missing.jpg"), image_id="image_1")
            path = _write(tmp, "big.jpg", JPEG)
            with patch.object(images_module, "MAX_IMAGE_BYTES", 8), self.assertRaisesRegex(ImageError, "larger than"):
                load_image(path, image_id="image_1")


class ModelInputTests(unittest.TestCase):
    def test_text_only_input_is_the_bare_json_string(self):
        payload = {"claim": "c", "requirements": {"max_checks": 2}}
        self.assertEqual(stage_input(payload), json.dumps(payload, indent=2, ensure_ascii=False))

    def test_images_follow_the_payload_each_after_its_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            images = load_images([_write(tmp, "a.jpg", JPEG), _write(tmp, "b.png", PNG)])
        message = stage_input({"claim": "c"}, images)
        self.assertEqual(len(message), 1)
        self.assertEqual(message[0]["role"], "user")
        parts = message[0]["content"]
        self.assertEqual([part["type"] for part in parts], ["input_text", "input_text", "input_image", "input_text", "input_image"])
        self.assertEqual(json.loads(parts[0]["text"]), {"claim": "c"})
        self.assertEqual((parts[1]["text"], parts[3]["text"]), ("[image_1]", "[image_2]"))
        self.assertTrue(parts[4]["image_url"].startswith("data:image/png;base64,"))

    def test_the_images_block_sits_after_the_claim_and_only_when_there_are_images(self):
        payload = {"reference_date": "2020-01-01", "claim": "c", "check": {}}
        self.assertIs(with_images(payload, None), payload)
        block = images_payload("research", ["image_1", "image_2"], searchable_ids=["image_2"])
        self.assertEqual(list(with_images(payload, block)), ["reference_date", "claim", "images", "check"])
        self.assertEqual(block["reverse_image_search"]["image_ids"], ["image_2"])
        self.assertNotIn("reverse_image_search", images_payload("research", ["image_1"]))
        self.assertEqual(list(with_images({"input_text": "t", "requirements": {}}, block))[1], "images")


class _Stage:
    def __init__(self, output) -> None:
        self.output = output
        self.calls: list[tuple[dict, tuple]] = []

    async def run(self, payload, images=()):
        self.calls.append((payload, tuple(images)))
        output = self.output(payload) if callable(self.output) else self.output
        return output, False


def _checker(**config) -> tuple[FactChecker, _Stage, _Stage, _Stage]:
    plan = InvestigationPlan(
        claim="c",
        checks=[VerificationCheck(aspect_id="a", question="Q?", rationale="r", search_queries=["q"])],
    )
    finding = AspectFinding(aspect_id="a", question="Q?", signal=EvidenceSignal.REFUTES, summary="s", confidence=0.9)
    report = FactCheckReport(claim="c", verdict=VeracityVerdict.REFUTED, verdict_confidence=0.9, justification="j")
    planner, researcher, judge = _Stage(plan), _Stage(finding), _Stage(report)
    checker = FactChecker(
        FactCheckRuntimeConfig(research_timeout_seconds=0, **config), planner=planner, researcher=researcher, judge=judge
    )
    return checker, planner, researcher, judge


class PipelineImageTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_stage_sees_the_images_and_is_told_how_to_use_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            images = load_images([_write(tmp, "a.jpg", JPEG)])
        checker, planner, researcher, judge = _checker()
        run = await checker.check_claim("c", images=images)

        for stage, name in ((planner, "plan"), (researcher, "research"), (judge, "judge")):
            payload, sent = stage.calls[0]
            self.assertEqual(sent, tuple(images))
            self.assertEqual(payload["images"]["ids"], ["image_1"])
            self.assertEqual(payload["images"]["instruction"], IMAGE_NOTICES[name])
        self.assertNotIn("reverse_image_search", researcher.calls[0][0]["images"])  # tool not enabled
        self.assertEqual([image.image_id for image in run.artifacts.images], ["image_1"])

    async def test_reverse_image_search_is_offered_for_images_with_a_public_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            images = load_images([_write(tmp, "a.jpg", JPEG), _write(tmp, "b.jpg", JPEG)])
        images[1] = images[1].model_copy(update={"public_url": "https://host.example/b.jpg"})
        checker, _, researcher, _ = _checker(reverse_image_search=True)
        await checker.check_claim("c", images=images)
        self.assertEqual(researcher.calls[0][0]["images"]["reverse_image_search"]["image_ids"], ["image_2"])

    async def test_a_text_only_claim_sends_no_images_block(self):
        checker, planner, researcher, judge = _checker(reverse_image_search=True)
        await checker.check_claim("c")
        for stage in (planner, researcher, judge):
            self.assertNotIn("images", stage.calls[0][0])
            self.assertEqual(stage.calls[0][1], ())

    async def test_claims_can_be_extracted_from_images_alone(self):
        result = ClaimExtractionResult(input_text="", claims=[])
        stage = _Stage(result)
        with tempfile.TemporaryDirectory() as tmp:
            images = load_images([_write(tmp, "post.png", PNG)])
        await ClaimExtractor(stage, max_claims=3).extract_claims("  ", images=images)
        payload, sent = stage.calls[0]
        self.assertEqual(payload["images"]["instruction"], IMAGE_NOTICES["extract_claims"])
        self.assertEqual(sent, tuple(images))
        with self.assertRaisesRegex(ValueError, "empty"):
            await ClaimExtractor(stage, max_claims=3).extract_claims("  ")


class BatchRunnerImageTests(unittest.TestCase):
    def test_image_names_resolve_against_the_image_dir(self):
        record = {"images": ["a.jpg", "https://x.example/b.jpg", "/abs/c.jpg"]}
        self.assertEqual(
            image_sources(record, 0, "images", Path("/data/imgs")),
            ["/data/imgs/a.jpg", "https://x.example/b.jpg", "/abs/c.jpg"],
        )
        self.assertEqual(image_sources({"claim": "c"}, 0, "images", Path(".")), [])
        with self.assertRaisesRegex(ValueError, "must be a list"):
            image_sources({"images": "a.jpg"}, 3, "images", Path("."))

    def _run(self, records, *, images: dict[str, bytes], extra=(), env=None):
        calls: list[tuple] = []

        class _Service:
            async def check_claim(self, claim, claim_id=None, claim_date=None, images=()):
                calls.append((claim, tuple(image.image_id for image in images), tuple(image.public_url for image in images)))
                report = FactCheckReport(claim=claim, verdict=VeracityVerdict.REFUTED, verdict_confidence=0.9, justification="j")
                return SimpleNamespace(report=report, artifacts=RunArtifacts(claim=claim, normalized_claim=claim))

        with tempfile.TemporaryDirectory() as tmp:
            for name, data in images.items():
                _write(tmp, name, data)
            input_path = Path(tmp) / "claims.json"
            input_path.write_text(json.dumps(records), encoding="utf-8")
            output = Path(tmp) / "out.json"
            args = build_batch_parser().parse_args(["--input", str(input_path), "--output", str(output), *extra])
            stderr = io.StringIO()
            environment = {"OPENAI_API_KEY": "k", "OPENAI_API_MODEL": "m", "SERPER_API_KEY": "s", "EXA_API_KEY": "e"}
            with (
                patch.dict("os.environ", {**environment, **(env or {})}, clear=False),
                patch("facticli.averitec_submission.build_fact_checker", return_value=_Service()) as build,
                redirect_stderr(stderr),
            ):
                code = asyncio.run(run_batch(args))
            self.config = build.call_args.args[0] if build.called else None
            manifest_path = Path(tmp) / "out.runinfo.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        return code, calls, manifest, stderr.getvalue()

    def test_each_claim_is_checked_with_its_images(self):
        records = [{"claim": "with", "images": ["a.jpg"]}, {"claim": "without"}]
        code, calls, manifest, _ = self._run(
            records, images={"a.jpg": JPEG}, extra=("--image-url-base", "https://host.example/i")
        )
        self.assertEqual(code, 0)
        self.assertEqual(
            sorted(calls), [("with", ("image_1",), ("https://host.example/i/a.jpg",)), ("without", (), ())]
        )
        self.assertEqual(manifest["claims"][0]["images"], 1)
        self.assertEqual(manifest["settings"]["image_url_base"], "https://host.example/i")
        # Searchable images turn reverse image search on without a flag.
        self.assertTrue(manifest["settings"]["reverse_image_search"])
        self.assertTrue(self.config.reverse_image_search)

    def test_reverse_image_search_stays_off_for_text_only_or_unsearchable_batches(self):
        no_keys = {"SERPER_API_KEY": "", "EXA_API_KEY": ""}
        code, _, manifest, _ = self._run([{"claim": "text only"}], images={}, env=no_keys)
        self.assertEqual((code, manifest["settings"]["reverse_image_search"]), (0, False))
        code, _, manifest, _ = self._run([{"claim": "c", "images": ["a.jpg"]}], images={"a.jpg": JPEG}, env=no_keys)
        self.assertEqual((code, manifest["settings"]["reverse_image_search"]), (0, False))  # local file, no URL base

    def test_searchable_images_need_the_keys_unless_the_search_is_turned_off(self):
        records = [{"claim": "c", "images": ["https://x.example/a.jpg"]}]
        code, calls, _, stderr = self._run(records, images={}, env={"SERPER_API_KEY": ""})
        self.assertEqual((code, calls), (2, []))
        self.assertIn("--no-reverse-image-search", stderr)
        with patch("facticli.averitec_submission.load_images", return_value=[]):
            code, _, manifest, _ = self._run(
                records, images={}, env={"SERPER_API_KEY": ""}, extra=("--no-reverse-image-search",)
            )
        self.assertEqual((code, manifest["settings"]["reverse_image_search"]), (0, False))

    def test_a_missing_image_stops_the_batch_before_the_first_claim(self):
        code, calls, _, stderr = self._run([{"claim": "c", "images": ["missing.jpg"]}], images={})
        self.assertEqual((code, calls), (2, []))
        self.assertIn("image file not found", stderr)


class CliImageTests(unittest.IsolatedAsyncioTestCase):
    async def test_reverse_image_search_needs_both_keys(self):
        args = build_parser().parse_args(["check", "c", "--reverse-image-search"])
        self.assertIsNone(build_parser().parse_args(["check", "c"]).reverse_image_search)  # decided by the images
        stderr = io.StringIO()
        env = {"OPENAI_API_KEY": "k", "OPENAI_API_MODEL": "m", "SERPER_API_KEY": "", "EXA_API_KEY": "e"}
        with patch.dict("os.environ", env, clear=False), redirect_stderr(stderr):
            self.assertEqual(await run_check_command(args), 2)
        self.assertIn("SERPER_API_KEY", stderr.getvalue())

    async def test_an_unusable_image_exits_before_any_model_call(self):
        args = build_parser().parse_args(["check", "c", "--image", "/nonexistent/x.jpg"])
        stderr = io.StringIO()
        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "k", "OPENAI_API_MODEL": "m"}, clear=False),
            patch("facticli.cli.build_fact_checker") as build,
            redirect_stderr(stderr),
        ):
            self.assertEqual(await run_check_command(args), 2)
        build.assert_not_called()
        self.assertIn("not found", stderr.getvalue())

    async def test_reverse_image_search_without_a_public_url_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "a.jpg", JPEG)
            args = build_parser().parse_args(["check", "c", "--image", path, "--reverse-image-search", "--json"])
            seen: list = []

            class _Service:
                async def check_claim(self, claim, progress=None, images=()):
                    seen.extend(images)
                    report = FactCheckReport(claim=claim, verdict=VeracityVerdict.REFUTED, verdict_confidence=0.9, justification="j")
                    return SimpleNamespace(report=report, plan=None, findings=[], artifacts=RunArtifacts(claim=claim, normalized_claim=claim))

            stderr = io.StringIO()
            env = {"OPENAI_API_KEY": "k", "OPENAI_API_MODEL": "m", "SERPER_API_KEY": "s", "EXA_API_KEY": "e"}
            with (
                patch.dict("os.environ", env, clear=False),
                patch("facticli.cli.build_fact_checker", return_value=_Service()),
                redirect_stderr(stderr),
                patch("sys.stdout", new=io.StringIO()),
            ):
                self.assertEqual(await run_check_command(args), 0)
        self.assertEqual([image.image_id for image in seen], ["image_1"])
        self.assertIn("needs a public image URL", stderr.getvalue())

    async def test_an_image_url_turns_reverse_image_search_on_and_a_local_file_does_not(self):
        configs: list = []

        class _Service:
            async def check_claim(self, claim, progress=None, images=()):
                report = FactCheckReport(claim=claim, verdict=VeracityVerdict.REFUTED, verdict_confidence=0.9, justification="j")
                return SimpleNamespace(report=report, plan=None, findings=[], artifacts=RunArtifacts(claim=claim, normalized_claim=claim))

        def build(config, artifacts_dir=None):
            configs.append(config)
            return _Service()

        with tempfile.TemporaryDirectory() as tmp:
            local = _write(tmp, "a.jpg", JPEG)
            env = {"OPENAI_API_KEY": "k", "OPENAI_API_MODEL": "m", "SERPER_API_KEY": "s", "EXA_API_KEY": "e"}
            with (
                patch.dict("os.environ", env, clear=False),
                patch("facticli.cli.build_fact_checker", side_effect=build),
                patch("facticli.cli.load_images", side_effect=lambda sources, url_base=None: [
                    images_module.ClaimImage(
                        image_id="image_1", source=sources[0], mime_type="image/jpeg", byte_size=1, sha256="0",
                        public_url=sources[0] if sources[0].startswith("http") else None,
                    )
                ] if sources else []),
                patch("sys.stdout", new=io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                for argv in (
                    ["check", "c", "--image", "https://x.example/a.jpg"],
                    ["check", "c", "--image", local],
                    ["check", "c"],
                    ["check", "c", "--image", "https://x.example/a.jpg", "--no-reverse-image-search"],
                ):
                    self.assertEqual(await run_check_command(build_parser().parse_args([*argv, "--json"])), 0)
        self.assertEqual([config.reverse_image_search for config in configs], [True, False, False, False])

    async def test_extract_claims_accepts_images_without_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = build_parser().parse_args(["extract-claims", "--image", _write(tmp, "post.png", PNG), "--json"])
            extractor = SimpleNamespace(calls=[])

            async def extract(text, images=()):
                extractor.calls.append((text, len(images)))
                return ClaimExtractionResult(input_text=text)

            extractor.extract_claims = extract
            with (
                patch.dict("os.environ", {"OPENAI_API_KEY": "k", "OPENAI_API_MODEL": "m"}, clear=False),
                patch("facticli.cli.build_claim_extractor", return_value=extractor),
                patch("sys.stdout", new=io.StringIO()),
            ):
                self.assertEqual(await run_extract_claims_command(args), 0)
        self.assertEqual(extractor.calls, [("", 1)])


if __name__ == "__main__":
    unittest.main()
