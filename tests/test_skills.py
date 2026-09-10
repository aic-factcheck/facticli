from __future__ import annotations

import unittest

from facticli.skills import get_skill, list_skills, load_skill_metadata, load_skill_prompt, parse_frontmatter


class SkillRegistryTests(unittest.TestCase):
    def test_list_skills_includes_core_and_extraction_skills(self):
        names = [skill.name for skill in list_skills()]
        self.assertEqual(names, ["plan", "research", "review", "judge", "single_agent", "extract_claims"])

    def test_descriptions_come_from_prompt_frontmatter(self):
        for skill in list_skills():
            meta = load_skill_metadata(skill.name)
            self.assertEqual(meta.get("name"), skill.name, skill.name)
            self.assertTrue(meta.get("description"), skill.name)
            self.assertEqual(skill.description, meta["description"])
            self.assertFalse(load_skill_prompt(skill.name).startswith("---"), skill.name)

    def test_parse_frontmatter_handles_missing_and_malformed_headers(self):
        self.assertEqual(parse_frontmatter("Plain body"), ({}, "Plain body"))
        meta, body = parse_frontmatter("---\nname: x\ndescription: \"Why\"\n---\n\nBody")
        self.assertEqual(meta, {"name": "x", "description": "Why"})
        self.assertEqual(body, "Body")
        malformed = "---\nno colon here\n---\nBody"
        self.assertEqual(parse_frontmatter(malformed), ({}, malformed))

    def test_prompts_are_lane_contracts(self):
        for name in ("plan", "research", "review", "judge", "single_agent"):
            prompt = load_skill_prompt(name)
            for section in ("# Purpose", "# Non-goals", "# Procedure", "# Failure handling", "# Output contract"):
                self.assertIn(section, prompt, f"{name} lacks {section}")

    def test_get_skill_unknown_raises(self):
        with self.assertRaises(KeyError):
            get_skill("nope")

    def test_extract_claims_prompt_is_loadable(self):
        prompt = load_skill_prompt("extract_claims")
        self.assertIn("decontextualized", prompt.lower())
        self.assertIn("atomic", prompt.lower())

    def test_review_prompt_is_loadable(self):
        prompt = load_skill_prompt("review")
        self.assertIn("follow-up", prompt.lower())
        self.assertIn("finalize", prompt.lower())


if __name__ == "__main__":
    unittest.main()
