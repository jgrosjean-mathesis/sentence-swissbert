import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from transformers import XLMRobertaTokenizer

import nh3

SEPARATOR = ' '

SKIPPED_MEDIA = {
    "srf Audio",
}


@dataclass
class SwissdoxArticle:
    id: str
    medium_name: str
    language: str
    head_xml: str
    content_xml: str
    content_clean: str = None

    def __str__(self):
        return f"{self.head_xml} ({self.language})"

    def to_txt(self) -> str:
        assert self.content_clean is not None
        s = []
        if self.content_clean:
            s.append(self.content_clean)
        text = ''.join(s)
        return text.strip()


class SwissdoxData:

    def __init__(self, tsv_path: Path):
        self.tsv_path = tsv_path
        self.cleaner = SwissdoxCleaner()

    def get_articles(self) -> Iterable[SwissdoxArticle]:
        seen_article_hashes = set()
        num_duplicates = 0
        num_filtered = 0
        with self.tsv_path.open() as f:
            csv.field_size_limit(1000000)
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if any(row[key] is None for key in ["id", "language", "head", "subhead", "content"]):
                    logging.info(f"Skipping row with missing values: {row['id']}")
                    continue
                article = SwissdoxArticle(
                    id=row["id"].strip(),
                    medium_name=row["medium_name"].strip(),
                    language=row["language"].strip(),
                    head_xml=row["head"].strip(),
                    content_xml=row["content"].strip(),
                )
                assert article.language in {"de", "fr", "it", "rm"}

                if article.medium_name in SKIPPED_MEDIA:
                    continue

                article_hash = hash(article.content_xml)
                if article_hash in seen_article_hashes:
                    num_duplicates += 1
                    continue
                seen_article_hashes.add(article_hash)
                article.content_clean = self.cleaner.clean(row["content"])

                yield article

        print(f"Skipped {num_duplicates} duplicates")
        print(f"Skipped {num_filtered} filtered articles")


class SwissdoxCleaner:

    def __init__(self):
        self.author_line_re = re.compile(r'<au>.*</au>|<ur>.*</ur>')
        self.separators_re = re.compile(r'<zt>|</zt>|<lg>|</lg>|<ka>|</ka>')
        self.paragraph_re = re.compile(r'<p>|</p>')
        self.link_start_re = re.compile(r'<a ')
        self.link_end_re = re.compile(r'</a>')
        self.sep_placeholder_re = re.compile(r'\[SEP]')
        self.double_sep_re = re.compile(rf'{SEPARATOR}\s*{SEPARATOR}')
        self.nbsp_re = re.compile(r'&nbsp;')
        self.amp_re = re.compile(r'&amp;')
        self.quot_re = re.compile(r'&quot;')
        self.lt_re = re.compile(r'&lt;')
        self.gt_re = re.compile(r'&gt;')

    def clean(self, xml: str) -> str:
        # Remove author lines
        xml = re.sub(self.author_line_re, '', xml)
        # Replace crossheadings, boxes and legends with </s>
        # Use intermediary sep symbol to avoid interference with bleach
        xml = re.sub(self.separators_re, '[SEP]', xml)
        # Add a space around hyperlinks
        xml = re.sub(self.link_start_re, ' <a ', xml)
        xml = re.sub(self.link_end_re, '</a> ', xml)
        # Replace <p> before bleach to avoid linebreaks
        xml = re.sub(self.paragraph_re, ' ', xml)
        text = nh3.clean(xml, tags=set())
        # Resolve common HTML entities
        text = re.sub(self.nbsp_re, ' ', text)
        text = re.sub(self.amp_re, '&', text)
        text = re.sub(self.quot_re, '"', text)
        text = re.sub(self.lt_re, '<', text)
        text = re.sub(self.gt_re, '>', text)
        text = text.replace('\xa0', ' ')  # nbsp
        text = text.replace('\xad', '')  # shy
        text = re.sub(self.sep_placeholder_re, SEPARATOR, text)
        # Remove duplicate separators
        text = re.sub(self.double_sep_re, SEPARATOR, text)
        text = re.sub(self.double_sep_re, SEPARATOR, text)
        text = text.replace("\n", " ")
        # Element should not be wrapped in separators
        if text.startswith(SEPARATOR):
            text = text[len(SEPARATOR):]
        if text.endswith(SEPARATOR):
            text = text[:-len(SEPARATOR)]
        text = text.strip()
        return text
        