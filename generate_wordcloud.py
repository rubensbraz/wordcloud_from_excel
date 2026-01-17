import logging
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
from typing import List, Dict, Set, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter
from wordcloud import WordCloud
from janome.tokenizer import Tokenizer
from matplotlib import font_manager
from nltk.corpus import stopwords


# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# --- WORDCLOUD CONFIGURATION ---
@dataclass
class CloudConfig:
    """Configuration schema for WordCloud generation.

    Attributes:
        language: Target language for NLP (e.g., 'english', 'japanese', 'portuguese').
        font_path: Absolute path to a CJK-supported font file (.ttf, .ttc).
        colormap: Matplotlib colormap theme (e.g., 'viridis', 'plasma', 'magma').
        width: Image width in pixels for high-definition output.
        height: Image height in pixels for high-definition output.
        max_words: Maximum number of words to display in the cloud.
        background_color: Background color for the generated image.
        stopword_file: Optional path to an external .txt file with custom stopwords.
    """

    language: str = "english"
    font_path: str = "C:/Windows/Fonts/msgothic.ttc"
    colormap: str = "viridis"
    width: int = 1920
    height: int = 1080
    max_words: int = 150
    background_color: str = "white"
    stopword_file: Optional[str] = None


# --- WORDCLOUD GENERATOR ---
class WordCloudArchitect:
    """A professional engine to generate high-quality Word Clouds from Excel data.

    This class orchestrates the entire flow from morphological analysis to
    high-fidelity visual rendering, supporting multilingual environments.
    """

    def __init__(self, config: CloudConfig):
        """Initializes the architect with configuration and NLP engines.

        Args:
            config: An instance of CloudConfig containing all environment settings.
        """
        self.config = config
        self.language_lower = config.language.lower()
        self.tokenizer_jp = Tokenizer() if self.language_lower == "japanese" else None

        # Ensure all necessary NLP resources are available
        self._ensure_nltk_resources()

        # Verify and handle font accessibility
        self._validate_font_assets()

        # Build the master set of filtered terms
        self.stop_set = self._initialize_stopwords()

    def _ensure_nltk_resources(self) -> None:
        """Safely verifies and downloads NLTK stopword corpora."""
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            logger.info("NLTK stopwords not found. Downloading...")
            nltk.download("stopwords", quiet=True)

    def _validate_font_assets(self) -> None:
        """Ensures the font path exists or falls back to system defaults."""
        font_p = Path(self.config.font_path)
        if not font_p.exists():
            fallback = font_manager.findfont(font_manager.FontProperties())
            logger.warning(
                f"Font not found at {self.config.font_path}. Falling back to: {fallback}"
            )
            self.config.font_path = fallback

    def _initialize_stopwords(self) -> Set[str]:
        """Combines NLTK, Japanese structural, and external stopwords.

        Returns:
            Set[str]: A unified, high-speed lookup set for word filtering.
        """
        master_set = set()

        # 1. Load NLTK stopwords for Western languages
        if self.language_lower != "japanese":
            try:
                master_set.update(stopwords.words(self.language_lower))
            except Exception as e:
                logger.warning(
                    f"NLTK language '{self.language_lower}' unavailable: {e}"
                )

        # 2. Add comprehensive Japanese structural stopwords (Resources Restored)
        if self.language_lower == "japanese":
            master_set.update(
                {
                    "する",
                    "ある",
                    "いる",
                    "これ",
                    "それ",
                    "あれ",
                    "どの",
                    "もの",
                    "こと",
                    "とき",
                    "よう",
                    "ほう",
                    "ため",
                    "から",
                    "まで",
                    "だけ",
                    "ほど",
                    "ばかり",
                    "なっ",
                    "でき",
                    "あり",
                    "いっ",
                    "おり",
                    "れる",
                    "られる",
                    "など",
                    "ため",
                }
            )

        # 3. Incorporate external custom file if provided
        if self.config.stopword_file:
            path = Path(self.config.stopword_file)
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        master_set.update(
                            {line.strip().lower() for line in f if line.strip()}
                        )
                except Exception as e:
                    logger.error(f"Error reading external stopword file: {e}")
            else:
                logger.error(f"Stopword file path does not exist: {path}")

        return master_set

    def tokenize(self, text: str) -> List[str]:
        """Segments raw text into semantically relevant tokens.

        Args:
            text: Raw input string from the data source.

        Returns:
            List[str]: Cleaned tokens filtered by POS and stopwords.
        """
        if not isinstance(text, str) or not text.strip():
            return []

        # Japanese Morphological Analysis
        if self.language_lower == "japanese":
            tokens = self.tokenizer_jp.tokenize(text.lower())
            words = []
            for token in tokens:
                pos = token.part_of_speech.split(",")[0]
                surface = token.surface.lower()
                # Extract only Nouns and Adjectives to ensure high-quality clouds
                if pos in ["名詞", "形容詞"] and surface not in self.stop_set:
                    if len(surface) > 1 and not surface.isdigit():
                        words.append(surface)
            return words

        # Western Regex-based Tokenization
        raw_words = re.findall(r"\b\w+\b", text.lower())
        return [
            w
            for w in raw_words
            if w not in self.stop_set and not w.isdigit() and len(w) > 1
        ]

    def generate_cloud(
        self, word_freq: Dict[str, int], title: str, output_path: Path
    ) -> None:
        """Renders the Word Cloud and saves it with professional resolution.

        Args:
            word_freq: Dictionary mapping words to their frequencies.
            title: Title text to display above the Word Cloud.
            output_path: Full Path object where the .png will be saved.
        """
        try:
            if not word_freq:
                logger.warning(
                    f"Skipping visualization for '{title}': Empty frequency data."
                )
                return

            # Build the cloud with user configuration
            wc = WordCloud(
                width=self.config.width,
                height=self.config.height,
                background_color=self.config.background_color,
                colormap=self.config.colormap,
                font_path=self.config.font_path,
                max_words=self.config.max_words,
                prefer_horizontal=0.8,
                relative_scaling=0.5,
            ).generate_from_frequencies(word_freq)

            # Matplotlib Rendering
            plt.figure(figsize=(16, 9))
            plt.imshow(wc, interpolation="bilinear")

            # Use specific font properties for CJK titles
            font_prop = font_manager.FontProperties(fname=self.config.font_path)
            plt.title(title, fontproperties=font_prop, fontsize=28, pad=25)
            plt.axis("off")

            # Save with 300 DPI for print/presentation quality
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            plt.close()
            logger.info(f"Cloud generated successfully: {output_path.name}")

        except Exception as e:
            logger.error(f"Failed to generate cloud for {title}: {e}")

    def run_pipeline(
        self, input_file: str, sheet: str, text_col: str, key_col: str, output_dir: str
    ) -> None:
        """Orchestrates the full data pipeline from Excel to Visual Report.

        Args:
            input_file: Path to the source .xlsx file.
            sheet: Name of the worksheet.
            text_col: Column name containing raw text.
            key_col: Column name for categorization.
            output_dir: Folder path for results.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        try:
            # Load and validate Excel data
            df = pd.read_excel(input_file, sheet_name=sheet)

            # Check if columns exist before processing
            required_cols = {text_col, key_col}
            if not required_cols.issubset(df.columns):
                missing = required_cols - set(df.columns)
                raise ValueError(f"Required columns missing in Excel: {missing}")

            logger.info(f"Data source loaded. Processing {len(df)} records.")
        except Exception as e:
            logger.error(f"Critical error loading data source: {e}")
            return

        all_statistics = []
        unique_keys = df[key_col].dropna().unique()

        # Iterate through unique segments
        for key in unique_keys:
            logger.info(f"Processing segment: {key}")
            subset = df[df[key_col] == key]

            # Clean text by removing NaN and joining
            combined_text = " ".join(subset[text_col].dropna().astype(str))

            # Tokenize and count
            words = self.tokenize(combined_text)
            counts = Counter(words)

            # Accumulate data for the statistical report
            for word, freq in counts.most_common(300):
                all_statistics.append(
                    {"Category": key, "Word": word, "Frequency": freq}
                )

            # Filename sanitization and cloud generation
            safe_name = re.sub(r'[\\/*?:"<>|]', "", str(key))
            img_file_path = out_path / f"cloud_{safe_name}.png"
            self.generate_cloud(dict(counts), f"Keyword: {key}", img_file_path)

        # Export consolidated statistical Excel report
        try:
            report_path = out_path / "analysis_report.xlsx"
            pd.DataFrame(all_statistics).to_excel(report_path, index=False)
            logger.info(f"Report exported: {report_path}")
            logger.info(f"[COMPLETED] All files saved in: '{output_dir}'")
        except Exception as e:
            logger.error(f"Failed to export statistical report: {e}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Theme configuration: Swap 'colormap' to 'magma', 'plasma', 'Spectral', etc
    # Supported languages: 'english', 'japanese', 'portuguese', 'spanish', 'french', etc
    wordcloud_config = CloudConfig(
        language="english",
        colormap="viridis",
        width=1920,
        height=1080,
        max_words=100,
        stopword_file="input/stopwords_en.txt",
    )

    # Initialize the architect
    architect = WordCloudArchitect(config=wordcloud_config)

    # Run the pipeline
    architect.run_pipeline(
        input_file="input/Textual analysis Gardens.xlsx",
        sheet="Database",
        text_col="Translated Text: English",
        key_col="Keyword English",
        output_dir="output",
    )
