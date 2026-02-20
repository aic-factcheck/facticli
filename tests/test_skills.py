from __future__ import annotations

import unittest

from facticli.skills import list_skills, load_skill_prompt


class SkillRegistryTests(unittest.TestCase):
    def test_list_skills_includes_core_and_extraction_skills(self):
        names = [skill.name for skill in list_skills()]
        self.assertEqual(names, ["plan", "research", "judge", "extract_claims"])

    def test_extract_claims_prompt_is_loadable(self):
        prompt = load_skill_prompt("extract_claims")
        self.assertIn("decontextualized", prompt.lower())
        self.assertIn("atomic", prompt.lower())


if __name__ == "__main__":
    unittest.main()

