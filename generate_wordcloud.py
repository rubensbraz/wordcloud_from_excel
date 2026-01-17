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
    """
    A professional-grade tool to generate Word Clouds from Excel data,
    supporting multilingual processing (English, Japanese, Portuguese, etc.)
    with NLTK and custom stopword integration.
    """

    def __init__(
        self, font_path: str, language: str = "english", stopwords_file: str = None
    ):
        """
        Initializes the architect with font settings, language, and stopword filters.

        Args:
            font_path (str): Path to a CJK-supported font file (.ttf, .ttc).
            language (str): Language name for NLTK (e.g., 'english', 'portuguese', 'japanese').
            stopwords_file (str, optional): Path to a .txt file containing
                                            custom stopwords (one per line).
        """
        self.language = language.lower()
        self.font_path = font_path
        self.tokenizer_jp = Tokenizer() if self.language == "japanese" else None

        # Ensure NLTK resources are available
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)

        # Initialize the master stopword set
        self.stop_set = self._initialize_stopwords(stopwords_file)

    def _initialize_stopwords(self, stopwords_file: str) -> Set[str]:
        """
        Combines NLTK, internal, and external stopwords.

        Returns:
            Set[str]: A unified set of words to be ignored.
        """
        master_set = set()

        # 1. Load NLTK stopwords (if the language is supported by NLTK)
        # Note: NLTK does not have a 'japanese' corpus, we handle it internally.
        if self.language != "japanese":
            try:
                master_set.update(stopwords.words(self.language))
            except Exception as e:
                print(f"Warning: NLTK language '{self.language}' not found. {e}")

        # 2. Internal Japanese structural stopwords
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
                    "いる",
                    "なっ",
                    "でき",
                    "あり",
                    "いっ",
                    "おり",
                    "れる",
                    "られる",
                }
            )

        # 3. External custom file
        if stopwords_file:
            master_set.update(self._load_external_stopwords(stopwords_file))

        return master_set

    def _load_external_stopwords(self, file_path: str) -> Set[str]:
        """Reads stopwords from an external text file."""
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
        """Removes illegal characters from filenames."""
        return re.sub(r'[\\/*?:"<>|]', "", filename)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes text based on the configured language.

        Returns:
            List[str]: Cleaned tokens (nouns/adjectives for JP, all valid words for others).
        """
        if not isinstance(text, str) or not text.strip():
            return []

        if self.language == "japanese":
            # Japanese Morphological Analysis Logic
            tokens = self.tokenizer_jp.tokenize(text)
            words = []
            for token in tokens:
                pos = token.part_of_speech.split(",")[0]
                surface = token.surface.lower()
                if pos in ["名詞", "形容詞"] and surface not in self.stop_set:
                    if len(surface) > 1 and not surface.isdigit():
                        words.append(surface)
            return words
        else:
            # Standard Western Language Logic
            # Regex extracts only alphanumeric words
            raw_words = re.findall(r"\b\w+\b", text.lower())
            return [
                w
                for w in raw_words
                if w not in self.stop_set and not w.isdigit() and len(w) > 1
            ]

    def generate_cloud(self, word_freq: Dict[str, int], title: str, output_path: str):
        """Generates and saves the WordCloud image."""
        try:
            if not word_freq:
                print(f"-> Skipping '{title}': Empty frequency dict.")
                return

            # Setup CJK font properties for Matplotlib
            font_prop = font_manager.FontProperties(fname=self.font_path)

            wc = WordCloud(
                width=1920,
                height=1080,
                background_color="white",
                colormap="viridis",
                font_path=self.font_path,
                max_words=60,
                prefer_horizontal=0.7,
            ).generate_from_frequencies(word_freq)

            plt.figure(figsize=(16, 9))
            plt.imshow(wc, interpolation="bilinear")
            plt.title(title, fontproperties=font_prop, fontsize=28, pad=25)
            plt.axis("off")
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"-> Cloud saved: {output_path}")
        except Exception as e:
            print(f"-> Error generating cloud for {title}: {e}")

    def run_pipeline(
        self, input_file: str, sheet: str, text_col: str, key_col: str, output_dir: str
    ):
        """Main execution flow: Load -> Process -> Save."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            df = pd.read_excel(input_file, sheet_name=sheet)
            print(f"-> Data loaded. Language set to: {self.language}")
        except Exception as e:
            print(f"-> Error loading Excel: {e}")
            return

        all_results = []
        unique_keys = df[key_col].dropna().unique()

        for key in unique_keys:
            print(f"Processing: {key}")
            subset = df[df[key_col] == key]
            text = " ".join(subset[text_col].dropna().astype(str))

            words = self.tokenize(text)
            counts = Counter(words)

            for word, freq in counts.most_common(300):
                all_results.append({"Keyword": key, "Word": word, "Frequency": freq})

            safe_name = self._sanitize_filename(str(key))
            img_path = os.path.join(output_dir, f"cloud_{safe_name}.png")
            self.generate_cloud(dict(counts), f"Analysis: {key}", img_path)

        # Save Report
        report_path = os.path.join(output_dir, "analysis_report.xlsx")
        pd.DataFrame(all_results).to_excel(report_path, index=False)
        print(f"\n[DONE] Check the '{output_dir}' folder.")


# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # 1. DEFINE LANGUAGE VARIABLE
    # Options: 'english', 'portuguese', 'spanish', 'french', 'japanese', etc
    SELECTED_LANGUAGE = "english"

    # 2. PATHS
    FONT = r"C:\Windows\Fonts\msgothic.ttc"
    # Choose the correct stopword file based on language if needed
    STOP_FILE = "input/stopwords_en.txt"

    # Initialize and Execute
    architect = WordCloudArchitect(
        font_path=FONT, language=SELECTED_LANGUAGE, stopwords_file=STOP_FILE
    )

    architect.run_pipeline(
        input_file="input/Textual analysis Gardens.xlsx",
        sheet="Database",
        text_col="Translated Text: English",
        key_col="Keyword English",
        output_dir="output",
    )
