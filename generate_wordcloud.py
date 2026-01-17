import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import nltk
from wordcloud import WordCloud
from collections import Counter
from janome.tokenizer import Tokenizer
from matplotlib import font_manager
from nltk.corpus import stopwords
from typing import List, Dict, Set, Any


class WordCloudArchitect:
    """A professional tool to generate high-quality Word Clouds from Excel data.

    This class handles multilingual text processing (English, Japanese, etc.)
    by integrating NLTK stopword sets and the Janome morphological analyzer.
    """

    def __init__(
        self, font_path: str, language: str = "english", stopwords_file: str = None
    ):
        """Initializes the architect with font settings and language logic.

        Args:
            font_path (str): Absolute path to a CJK-supported font file (.ttf, .ttc).
            language (str): Target language for NLTK (e.g., 'english', 'japanese').
            stopwords_file (str, optional): Path to an external .txt stopword file.
        """
        self.language = language.lower()
        self.font_path = font_path
        self.tokenizer_jp = Tokenizer() if self.language == "japanese" else None

        # Ensure NLTK resources are available locally
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)

        # Build the master set of words to ignore
        self.stop_set = self._initialize_stopwords(stopwords_file)

    def _initialize_stopwords(self, stopwords_file: str) -> Set[str]:
        """Combines multiple stopword sources into a single optimized set.

        Args:
            stopwords_file (str): Path to the user-defined stopword file.

        Returns:
            Set[str]: A unified set containing NLTK, internal, and external stopwords.
        """
        master_set = set()

        # 1. Load NLTK stopwords for non-Japanese languages
        if self.language != "japanese":
            try:
                master_set.update(stopwords.words(self.language))
            except Exception as e:
                print(f"Warning: NLTK language '{self.language}' unavailable. {e}")

        # 2. Add structural Japanese stopwords
        if self.language == "japanese":
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
                }
            )

        # 3. Incorporate external custom file
        if stopwords_file:
            master_set.update(self._load_external_stopwords(stopwords_file))

        return master_set

    def _load_external_stopwords(self, file_path: str) -> Set[str]:
        """Reads custom stopwords from a local text file.

        Args:
            file_path (str): Path to the .txt file (one word per line).

        Returns:
            Set[str]: Unique stopwords extracted from the file.
        """
        try:
            if not os.path.exists(file_path):
                print(f"Warning: Stopwords file '{file_path}' not found.")
                return set()
            with open(file_path, "r", encoding="utf-8") as f:
                return {line.strip().lower() for line in f if line.strip()}
        except Exception as e:
            print(f"Error loading external stopwords: {e}")
            return set()

    def _sanitize_filename(self, filename: str) -> str:
        """Removes reserved OS characters from a string to create a safe filename.

        Args:
            filename (str): The raw string to be sanitized.

        Returns:
            str: A clean string safe for file creation.
        """
        return re.sub(r'[\\/*?:"<>|]', "", filename)

    def tokenize(self, text: str) -> List[str]:
        """Segments raw text into a list of semantically relevant tokens.

        Args:
            text (str): The raw input string from the data source.

        Returns:
            List[str]: A list of cleaned words, filtered by language rules.
        """
        if not isinstance(text, str) or not text.strip():
            return []

        if self.language == "japanese":
            # Apply morphological analysis for Japanese
            tokens = self.tokenizer_jp.tokenize(text)
            words = []
            for token in tokens:
                pos = token.part_of_speech.split(",")[0]
                surface = token.surface.lower()
                # Extract only nouns and adjectives not in the stopword set
                if pos in ["名詞", "形容詞"] and surface not in self.stop_set:
                    if len(surface) > 1 and not surface.isdigit():
                        words.append(surface)
            return words
        else:
            # Use regex tokenization for Western languages
            raw_words = re.findall(r"\b\w+\b", text.lower())
            return [
                w
                for w in raw_words
                if w not in self.stop_set and not w.isdigit() and len(w) > 1
            ]

    def generate_cloud(self, word_freq: Dict[str, int], title: str, output_path: str):
        """Creates a WordCloud image and saves it to the specified directory.

        Args:
            word_freq (Dict[str, int]): Dictionary mapping words to frequencies.
            title (str): The title text to display above the cloud.
            output_path (str): Full path where the .png image will be saved.
        """
        try:
            if not word_freq:
                print(f"-> Skipping '{title}': Empty frequency data.")
                return

            if not os.path.exists(self.font_path):
                raise FileNotFoundError(f"Font file missing at: {self.font_path}")

            # Define font properties for CJK title support
            font_prop = font_manager.FontProperties(fname=self.font_path)

            # Build the cloud with high resolution and specific colormap
            wc = WordCloud(
                width=1920,
                height=1080,
                background_color="white",
                colormap="viridis",
                font_path=self.font_path,
                max_words=60,
                prefer_horizontal=0.7,
            ).generate_from_frequencies(word_freq)

            # Render plot using Matplotlib
            plt.figure(figsize=(16, 9))
            plt.imshow(wc, interpolation="bilinear")
            plt.title(title, fontproperties=font_prop, fontsize=28, pad=25)
            plt.axis("off")

            # Save output with 300 DPI for professional quality
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            plt.close()  # Clear memory
            print(f"-> Output generated: {output_path}")

        except Exception as e:
            print(f"-> Critical failure in Cloud Generation for {title}: {e}")

    def run_pipeline(
        self, input_file: str, sheet: str, text_col: str, key_col: str, output_dir: str
    ):
        """Orchestrates the full flow from Excel loading to visual export.

        Args:
            input_file (str): Path to the source .xlsx file.
            sheet (str): Name of the worksheet to process.
            text_col (str): Column name containing the raw text.
            key_col (str): Column name for keyword categorization.
            output_dir (str): Folder path for storing results.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            df = pd.read_excel(input_file, sheet_name=sheet)
            print(f"-> System ready. Mode: {self.language.upper()}")
        except Exception as e:
            print(f"-> Failed to load data source: {e}")
            return

        all_results = []
        unique_keys = df[key_col].dropna().unique()

        # Iterate through unique categories
        for key in unique_keys:
            print(f"Processing segment: {key}")
            subset = df[df[key_col] == key]
            combined_text = " ".join(subset[text_col].dropna().astype(str))

            # Clean and count words
            words = self.tokenize(combined_text)
            counts = Counter(words)

            # Build the frequency report dataset
            for word, freq in counts.most_common(300):
                all_results.append({"Keyword": key, "Word": word, "Frequency": freq})

            # Create sanitized filename and generate visualization
            safe_name = self._sanitize_filename(str(key))
            img_path = os.path.join(output_dir, f"cloud_{safe_name}.png")
            self.generate_cloud(dict(counts), f"Analysis: {key}", img_path)

        # Export the consolidated statistical report
        report_path = os.path.join(output_dir, "analysis_report.xlsx")
        pd.DataFrame(all_results).to_excel(report_path, index=False)
        print(f"\n[COMPLETE] All files saved in: '{output_dir}'")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Language Configuration: 'english', 'portuguese', 'japanese', etc.
    SELECTED_LANGUAGE = "english"

    # System Paths
    SYSTEM_FONT = r"C:\Windows\Fonts\msgothic.ttc"
    STOPWORDS_PATH = "input/stopwords_en.txt"

    # Application Instance
    architect = WordCloudArchitect(
        font_path=SYSTEM_FONT, language=SELECTED_LANGUAGE, stopwords_file=STOPWORDS_PATH
    )

    # Process Data
    architect.run_pipeline(
        input_file="input/Textual analysis Gardens.xlsx",
        sheet="Database",
        text_col="Translated Text: English",
        key_col="Keyword English",
        output_dir="output",
    )
