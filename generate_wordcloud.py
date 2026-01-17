import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from wordcloud import WordCloud
from collections import Counter
from janome.tokenizer import Tokenizer
from matplotlib import font_manager
from typing import List, Dict, Set, Any


class WordCloudArchitect:
    """
    A professional-grade tool to generate Word Clouds from Excel data,
    supporting English and Japanese processing with custom stopword support.
    """

    def __init__(self, font_path: str, stopwords_file: str = None):
        """
        Initializes the architect with font settings and stopword filters.

        Args:
            font_path (str): Path to a CJK-supported font file (.ttf, .ttc).
            stopwords_file (str, optional): Path to a .txt file containing
                                            custom stopwords (one per line).
        """
        self.tokenizer = Tokenizer()
        self.font_path = font_path

        # Default internal Japanese stopwords (functional particles and verbs)
        self.stopwords = {
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
            "よう",
            "あり",
            "いっ",
            "おり",
            "れる",
            "られる",
        }

        # Load external stopwords if provided
        if stopwords_file:
            self.stopwords.update(self._load_external_stopwords(stopwords_file))

    def _load_external_stopwords(self, file_path: str) -> Set[str]:
        """
        Reads stopwords from an external text file.

        Args:
            file_path (str): Path to the .txt file.

        Returns:
            Set[str]: A set of unique stopwords.
        """
        try:
            if not os.path.exists(file_path):
                print(
                    f"Warning: Stopwords file '{file_path}' not found. Using defaults."
                )
                return set()

            with open(file_path, "r", encoding="utf-8") as f:
                # Read lines, strip whitespace, and filter out empty lines
                return {line.strip().lower() for line in f if line.strip()}
        except Exception as e:
            print(f"Error loading stopwords: {e}")
            return set()

    def _sanitize_filename(self, filename: str) -> str:
        """
        Removes illegal characters from filenames to prevent OS errors.

        Args:
            filename (str): The raw string to be used as a filename.

        Returns:
            str: A clean, safe filename.
        """
        # Remove characters like / \ : * ? " < > |
        return re.sub(r'[\\/*?:"<>|]', "", filename)

    def load_excel_data(self, file_path: str, sheet_name: str) -> pd.DataFrame:
        """
        Loads data from a local Excel file with error handling.

        Args:
            file_path (str): Path to the .xlsx file.
            sheet_name (str): The worksheet name.

        Returns:
            pd.DataFrame: The loaded data or empty DataFrame on failure.
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File '{file_path}' not found.")

            df = pd.read_excel(file_path, sheet_name=sheet_name)
            print(f"-> Successfully loaded {len(df)} rows from '{sheet_name}'.")
            return df
        except Exception as e:
            print(f"-> Error loading Excel file: {e}")
            return pd.DataFrame()

    def tokenize_japanese(self, text: str) -> List[str]:
        """
        Tokenizes Japanese text into filtered nouns and adjectives.

        Args:
            text (str): Raw Japanese string.

        Returns:
            List[str]: List of semantically relevant words.
        """
        if not isinstance(text, str) or not text.strip():
            return []

        tokens = self.tokenizer.tokenize(text)
        words = []
        for token in tokens:
            # part_of_speech returns 'POS,Sub-POS,*,*'
            pos = token.part_of_speech.split(",")[0]
            surface = token.surface.lower()

            # Filter logic: Only Nouns/Adjectives, not in stopwords, length > 1
            if pos in ["名詞", "形容詞"] and surface not in self.stopwords:
                # Remove purely numeric strings and punctuation
                if len(surface) > 1 and not surface.isdigit():
                    words.append(surface)
        return words

    def generate_cloud(
        self,
        word_freq: Dict[str, int],
        title: str,
        output_filename: str,
        max_words: int = 60,
    ) -> None:
        """
        Generates, renders, and saves a Word Cloud image.

        Args:
            word_freq (Dict[str, int]): Dictionary of word frequencies.
            title (str): The title displayed on the plot.
            output_filename (str): The path to save the .png file.
            max_words (int): Limit of words in the visualization.
        """
        try:
            if not word_freq:
                print(f"-> Skipping '{title}': No words to process.")
                return

            if not os.path.exists(self.font_path):
                raise FileNotFoundError(f"Font not found at: {self.font_path}")

            # Font property for Japanese title rendering in Matplotlib
            jp_font_prop = font_manager.FontProperties(fname=self.font_path)

            wc = WordCloud(
                width=1920,
                height=1080,
                background_color="white",
                colormap="viridis",
                font_path=self.font_path,
                max_words=max_words,
                prefer_horizontal=0.7,
            ).generate_from_frequencies(word_freq)

            plt.figure(figsize=(16, 9))
            plt.imshow(wc, interpolation="bilinear")
            plt.title(title, fontproperties=jp_font_prop, fontsize=28, pad=25)
            plt.axis("off")

            # Final save with high quality
            plt.savefig(output_filename, bbox_inches="tight", dpi=300)
            plt.close()  # Critical to free memory
            print(f"-> Cloud saved: {output_filename}")

        except Exception as e:
            print(f"-> Critical error generating cloud for {title}: {e}")

    def run_pipeline(
        self,
        input_file: str,
        sheet_name: str,
        column_text: str,
        column_keyword: str,
        output_folder: str,
        output_excel_name: str,
    ):
        """
        Full orchestration of the analysis workflow.
        """
        # Ensure output directory exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created output folder: {output_folder}")

        df = self.load_excel_data(input_file, sheet_name)
        if df.empty:
            return

        all_report_data = []
        unique_keywords = df[column_keyword].dropna().unique()

        for keyword in unique_keywords:
            print(f"Processing keyword: {keyword}")

            # Filter and combine text
            subset = df[df[column_keyword] == keyword]
            combined_text = " ".join(subset[column_text].dropna().astype(str))

            # Tokenize and count
            words = self.tokenize_japanese(combined_text)
            counts = Counter(words)

            # Build report data (Top 300)
            for word, freq in counts.most_common(300):
                all_report_data.append(
                    {"Keyword": keyword, "Word": word, "Frequency": freq}
                )

            # Prepare image filename
            safe_keyword = self._sanitize_filename(str(keyword))
            img_path = os.path.join(output_folder, f"cloud_{safe_keyword}.png")

            # Generate visual
            self.generate_cloud(dict(counts), f"Keyword: {keyword}", img_path)

        # Save Excel Report
        try:
            results_df = pd.DataFrame(all_report_data)
            excel_path = os.path.join(output_folder, output_excel_name)
            results_df.to_excel(excel_path, index=False)
            print(f"\n[SUCCESS] Pipeline finished. Report saved at: {excel_path}")
        except Exception as e:
            print(f"[ERROR] Could not save Excel report: {e}")


if __name__ == "__main__":
    # --- CONFIGURATION ---
    # 1. Update with your valid system font path
    # Windows example: r"C:\Windows\Fonts\msgothic.ttc"
    MY_FONT = r"C:\Windows\Fonts\msgothic.ttc"

    # 2. Path to your custom stopwords file (optional)
    MY_STOPWORDS = "input/stopwords.txt"

    # Initialize Architect
    architect = WordCloudArchitect(font_path=MY_FONT, stopwords_file=MY_STOPWORDS)

    # Execute Pipeline
    architect.run_pipeline(
        input_file="input/Textual analysis Gardens.xlsx",
        sheet_name="Database",
        column_text="Original Text: Japanese",
        column_keyword="Keyword",
        output_folder="output",
        output_excel_name="final_analysis_report.xlsx",
    )
