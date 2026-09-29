"""Integration checks for the editable report workflow, without model training."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import build_report
from pypdf import PdfReader


class EditableReportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'section.md'
        self.table = self.root/'table.csv'
        self.output = self.root/'report.pdf'
        self.config = self.root/'report.json'
        self.config.write_text(json.dumps({'output':'report.pdf', 'sections':[{'file':'section.md'}]}))
        self.source.write_text('# Editable report\n\nOriginal wording.\n\n'
                               '::: table table.csv widths=83,83\nEditable caption.\n:::\n')
        self.table.write_text('Source,WQL\nControl,0.695413\n')

    def test_manual_text_and_table_edits_reach_pdf_without_changing_sources(self):
        build_report.build(self.config)
        self.source.write_text(self.source.read_text().replace('Original wording.',
            '**Revised wording** with 5 < 10 & a literal sign.\n\n- First item\n- Second item'))
        self.table.write_text(self.table.read_text().replace('0.695413', '0.123456'))
        before = (self.source.read_bytes(), self.table.read_bytes())
        build_report.build(self.config)
        text = '\n'.join(p.extract_text() for p in PdfReader(self.output).pages)
        self.assertIn('Revised wording', text)
        self.assertIn('5 < 10 & a literal sign', text)
        self.assertIn('Second item', text)
        self.assertIn('0.123456', text)
        self.assertNotIn('Original wording', text)
        self.assertEqual(before, (self.source.read_bytes(), self.table.read_bytes()))

    def test_bad_source_keeps_last_successful_pdf_and_identifies_file(self):
        build_report.build(self.config)
        previous = self.output.read_bytes()
        self.source.write_text('::: table absent.csv\nCaption\n:::\n')
        with self.assertRaisesRegex(ValueError, r'section.md:1: Missing table file'):
            build_report.build(self.config)
        self.assertEqual(previous, self.output.read_bytes())
        self.source.write_text('::: note\nUnclosed block\n')
        with self.assertRaisesRegex(ValueError, 'unclosed'):
            build_report.build(self.config)
        self.assertEqual(previous, self.output.read_bytes())

    def test_shell_entry_point_works_outside_repo_and_respects_output_override(self):
        alternate = self.root/'draft.pdf'
        result = subprocess.run([str(ROOT/'scripts/build_report.sh'), '--config', str(self.config),
                                 '--output', str(alternate)], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(alternate.is_file())
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
