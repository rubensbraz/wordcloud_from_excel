import os
import re
import logging
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

# --- CONFIGURATION LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class CloudConfig:
    """Configuration schema for WordCloud generation.

    Attributes:
        language: Target language for NLP processing.
        font_path: Path to the .ttf or .ttc font file.
        colormap: Matplotlib colormap theme (e.g., 'viridis', 'plasma', 'magma').
        width: Image width in pixels.
        height: Image height in pixels.
        max_words: Limit of words in the cloud.
        background_color: Canvas color.
    """

    language: str = "english"
    font_path: str = "C:/Windows/Fonts/msgothic.ttc"
    colormap: str = "viridis"
    width: int = 1920
    height: int = 1080
    max_words: int = 100
    background_color: str = "white"
    stopword_file: Optional[str] = None


class WordCloudArchitect:
    """High-performance engine for multilingual WordCloud synthesis."""

    def __init__(self, config: CloudConfig):
        """Initializes the architect with a configuration object.

        Args:
            config: An instance of CloudConfig with all parameters.
        """
        self.config = config
        self.tokenizer_jp = (
            Tokenizer() if config.language.lower() == "japanese" else None
        )

        # Ensure NLTK resources are available
        self._prepare_nltk()

        # Build optimized stopword cache
        self.stop_set = self._initialize_stopwords()

    def _prepare_nltk(self) -> None:
        """Downloads necessary NLTK corpora safely."""
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            nltk.download("stopwords", quiet=True)

    def _initialize_stopwords(self) -> Set[str]:
        """Merges multiple stopword sources into a high-speed lookup set.

        Returns:
            A set of strings containing all filtered words.
        """
        master_set = set()
        lang = self.config.language.lower()

        # 1. NLTK Base
        if lang != "japanese":
            try:
                master_set.update(stopwords.words(lang))
            except Exception as e:
                logger.warning(
                    f"NLTK language '{lang}' not found. Using default. Error: {e}"
                )

        # 2. Hardcoded Japanese structural noise
        if lang == "japanese":
            master_set.update(
                {"する", "ある", "いる", "これ", "それ", "もの", "こと", "とき"}
            )

        # 3. External File
        if self.config.stopword_file:
            path = Path(self.config.stopword_file)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    master_set.update(
                        {line.strip().lower() for line in f if line.strip()}
                    )
            else:
                logger.error(f"Stopword file not found at: {path}")

        return master_set

    def tokenize(self, text: str) -> List[str]:
        """Performs morphological analysis or regex tokenization.

        Args:
            text: Raw input string.

        Returns:
            List of processed tokens (nouns/adjectives).
        """
        if not isinstance(text, str) or not text.strip():
            return []

        # Sanitize basic noise
        text = text.lower()

        if self.config.language.lower() == "japanese":
            tokens = self.tokenizer_jp.tokenize(text)
            return [
                t.surface
                for t in tokens
                if t.part_of_speech.split(",")[0] in ["名詞", "形容詞"]
                and t.surface not in self.stop_set
                and len(t.surface) > 1
            ]

        # Western Tokenization
        words = re.findall(r"\b\w+\b", text)
        return [
            w
            for w in words
            if w not in self.stop_set and not w.isdigit() and len(w) > 1
        ]

    def generate_cloud(
        self, word_freq: Dict[str, int], title: str, output_path: Path
    ) -> None:
        """Renders the final visualization with error handling.

        Args:
            word_freq: Frequency mapping of tokens.
            title: Title to be displayed on the plot.
            output_path: Destination Path object for the PNG.
        """
        try:
            if not word_freq:
                logger.warning(f"No data available for: {title}")
                return

            if not Path(self.config.font_path).exists():
                raise FileNotFoundError(f"Font missing: {self.config.font_path}")

            # WordCloud Object Configuration
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
            plt.figure(figsize=(20, 10))
            plt.imshow(wc, interpolation="bilinear")

            # Font Support for Title (CJK)
            font_prop = font_manager.FontProperties(fname=self.config.font_path)
            plt.title(title, fontproperties=font_prop, fontsize=30, pad=30)
            plt.axis("off")

            # Save with high DPI for print quality
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            plt.close()
            logger.info(f"Visual exported successfully: {output_path.name}")

        except Exception as e:
            logger.error(f"Critical rendering error for {title}: {e}")

    def run_pipeline(
        self, input_file: str, sheet: str, text_col: str, key_col: str, output_dir: str
    ):
        """Executes the end-to-end extraction and visualization flow.

        Args:
            input_file: Source Excel path.
            sheet: Target worksheet name.
            text_col: Column containing textual data.
            key_col: Column for segmentation/category.
            output_dir: Target folder for artifacts.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        try:
            df = pd.read_excel(input_file, sheet_name=sheet)
            logger.info(f"Source loaded: {len(df)} rows found.")
        except Exception as e:
            logger.error(f"Failed to load Excel: {e}")
            return

        report_data = []
        unique_keys = df[key_col].dropna().unique()

        for key in unique_keys:
            logger.info(f"Processing segment: {key}")
            subset = df[df[key_col] == key]
            combined_text = " ".join(subset[text_col].dropna().astype(str))

            tokens = self.tokenize(combined_text)
            counts = Counter(tokens)

            # Statistical data for Excel report
            for word, freq in counts.most_common(300):
                report_data.append({"Category": key, "Term": word, "Frequency": freq})

            # Image generation
            safe_name = re.sub(r'[\\/*?:"<>|]', "", str(key))
            img_file = out_path / f"cloud_{safe_name}.png"
            self.generate_cloud(dict(counts), f"Analysis: {key}", img_file)

        # Export Excel Report
        try:
            report_df = pd.DataFrame(report_data)
            report_df.to_excel(out_path / "statistical_report.xlsx", index=False)
            logger.info("Pipeline complete. All artifacts saved.")
        except Exception as e:
            logger.error(f"Failed to save statistical report: {e}")


# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # 1. Define the "Theme" and Visual Settings
    # Matplotlib Colormaps: 'viridis', 'magma', 'inferno', 'plasma', 'coolwarm', 'Spectral', etc
    my_config = CloudConfig(
        language="english",
        colormap="plasma",  # Change this to any Matplotlib theme
        width=1920,
        height=1080,
        max_words=150,
        stopword_file="input/stopwords_en.txt",
    )

    # 2. Initialize Architect
    architect = WordCloudArchitect(config=my_config)

    # 3. Execute
    architect.run_pipeline(
        input_file="input/Textual analysis Gardens.xlsx",
        sheet="Database",
        text_col="Translated Text: English",
        key_col="Keyword English",
        output_dir="output",
    )
