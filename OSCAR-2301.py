# coding=utf-8
# Copyright 2023 The OSCAR Project Authors, Inria and DFKI GmbH.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""OSCAR The Open Super-large Crawled Aggregated coRpus."""


import collections
import io
import zstandard
import json

from dataclasses import dataclass

import datasets


logger = datasets.logging.get_logger(__name__)


@dataclass
class Identification:
    label: str
    prob: float


_DESCRIPTION = """\
The Open Super-large Crawled Aggregated coRpus is a huge multilingual corpus \
obtained by language classification and filtering of the Common Crawl corpus \
using the Ungoliant architecture.\
"""

_URL = "https://oscar-project.org"

_LICENSE = """
    These data are released under this licensing scheme
    We do not own any of the text from which these data has been extracted.
    We license the actual packaging of these data under the Creative Commons CC0 license \
    (\"no rights reserved\") http://creativecommons.org/publicdomain/zero/1.0/
    To the extent possible under law, Inria has waived all copyright \
    and related or neighboring rights to OSCAR
    This work is published from: France.
    Should you consider that our data contains material that is owned by you \
    and should therefore not be reproduced here, please:
    * Clearly identify yourself, with detailed contact data such as an address, \
    telephone number or email address at which you can be contacted.
    * Clearly identify the copyrighted work claimed to be infringed.
    * Clearly identify the material that is claimed to be infringing and \
    information reasonably sufficient to allow us to locate the material.
    We will comply to legitimate requests by removing the affected sources \
    from the next release of the corpus. \
"""

_CITATION = """\
@ARTICLE{2022arXiv221210440J,
       author = {{Jansen}, Tim and {Tong}, Yangling and {Zevallos}, Victoria and {Ortiz Suarez}, Pedro},
        title = "{Perplexed by Quality: A Perplexity-based Method for Adult and Harmful Content Detection in Multilingual Heterogeneous Web Data}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computation and Language},
         year = 2022,
        month = dec,
          eid = {arXiv:2212.10440},
        pages = {arXiv:2212.10440},
          doi = {10.48550/arXiv.2212.10440},
archivePrefix = {arXiv},
       eprint = {2212.10440},
 primaryClass = {cs.CL},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022arXiv221210440J},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
@inproceedings{abadji-etal-2022-towards,
    title = "Towards a Cleaner Document-Oriented Multilingual Crawled Corpus",
    author = "Abadji, Julien  and
      Ortiz Suarez, Pedro  and
      Romary, Laurent  and
      Sagot, Beno{\^\i}t",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.463",
    pages = "4344--4355",
    abstract = "The need for large corpora raw corpora has dramatically increased in recent years with the introduction of transfer learning and semi-supervised learning methods to Natural Language Processing. And while there have been some recent attempts to manually curate the amount of data necessary to train large language models, the main way to obtain this data is still through automatic web crawling. In this paper we take the existing multilingual web corpus OSCAR and its pipeline Ungoliant that extracts and classifies data from Common Crawl at the line level, and propose a set of improvements and automatic annotations in order to produce a new document-oriented version of OSCAR that could prove more suitable to pre-train large generative language models as well as hopefully other applications in Natural Language Processing and Digital Humanities.",
}
@inproceedings{AbadjiOrtizSuarezRomaryetal.2021,
  author    = {Julien Abadji and Pedro Javier Ortiz Su{\'a}rez and Laurent Romary and Beno{\^i}t Sagot},
  title     = {Ungoliant: An optimized pipeline for the generation of a very large-scale multilingual web corpus},
  series = {Proceedings of the Workshop on Challenges in the Management of Large Corpora (CMLC-9) 2021. Limerick, 12 July 2021 (Online-Event)},
  editor    = {Harald L{\"u}ngen and Marc Kupietz and Piotr Bański and Adrien Barbaresi and Simon Clematide and Ines Pisetta},
  publisher = {Leibniz-Institut f{\"u}r Deutsche Sprache},
  address   = {Mannheim},
  doi       = {10.14618/ids-pub-10468},
  url       = {https://nbn-resolving.org/urn:nbn:de:bsz:mh39-104688},
  pages     = {1 -- 9},
  year      = {2021},
  abstract  = {Since the introduction of large language models in Natural Language Processing, large raw corpora have played a crucial role in Computational Linguistics. However, most of these large raw corpora are either available only for English or not available to the general public due to copyright issues. Nevertheless, there are some examples of freely available multilingual corpora for training Deep Learning NLP models, such as the OSCAR and Paracrawl corpora. However, they have quality issues, especially for low-resource languages. Moreover, recreating or updating these corpora is very complex. In this work, we try to reproduce and improve the goclassy pipeline used to create the OSCAR corpus. We propose a new pipeline that is faster, modular, parameterizable, and well documented. We use it to create a corpus similar to OSCAR but larger and based on recent data. Also, unlike OSCAR, the metadata information is at the document level. We release our pipeline under an open source license and publish the corpus under a research-only license.},
  language  = {en}
}
@article{kreutzer-etal-2022-quality,
    title = "Quality at a Glance: An Audit of Web-Crawled Multilingual Datasets",
    author = {Kreutzer, Julia  and
      Caswell, Isaac  and
      Wang, Lisa  and
      Wahab, Ahsan  and
      van Esch, Daan  and
      Ulzii-Orshikh, Nasanbayar  and
      Tapo, Allahsera  and
      Subramani, Nishant  and
      Sokolov, Artem  and
      Sikasote, Claytone  and
      Setyawan, Monang  and
      Sarin, Supheakmungkol  and
      Samb, Sokhar  and
      Sagot, Beno{\^\i}t  and
      Rivera, Clara  and
      Rios, Annette  and
      Papadimitriou, Isabel  and
      Osei, Salomey  and
      Suarez, Pedro Ortiz  and
      Orife, Iroro  and
      Ogueji, Kelechi  and
      Rubungo, Andre Niyongabo  and
      Nguyen, Toan Q.  and
      M{\"u}ller, Mathias  and
      M{\"u}ller, Andr{\'e}  and
      Muhammad, Shamsuddeen Hassan  and
      Muhammad, Nanda  and
      Mnyakeni, Ayanda  and
      Mirzakhalov, Jamshidbek  and
      Matangira, Tapiwanashe  and
      Leong, Colin  and
      Lawson, Nze  and
      Kudugunta, Sneha  and
      Jernite, Yacine  and
      Jenny, Mathias  and
      Firat, Orhan  and
      Dossou, Bonaventure F. P.  and
      Dlamini, Sakhile  and
      de Silva, Nisansa  and
      {\c{C}}abuk Ball{\i}, Sakine  and
      Biderman, Stella  and
      Battisti, Alessia  and
      Baruwa, Ahmed  and
      Bapna, Ankur  and
      Baljekar, Pallavi  and
      Azime, Israel Abebe  and
      Awokoya, Ayodele  and
      Ataman, Duygu  and
      Ahia, Orevaoghene  and
      Ahia, Oghenefego  and
      Agrawal, Sweta  and
      Adeyemi, Mofetoluwa},
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "10",
    year = "2022",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2022.tacl-1.4",
    doi = "10.1162/tacl_a_00447",
    pages = "50--72",
    abstract = "With the success of large-scale pre-training and multilingual modeling in Natural Language Processing (NLP), recent years have seen a proliferation of large, Web-mined text datasets covering hundreds of languages. We manually audit the quality of 205 language-specific corpora released with five major public datasets (CCAligned, ParaCrawl, WikiMatrix, OSCAR, mC4). Lower-resource corpora have systematic issues: At least 15 corpora have no usable text, and a significant fraction contains less than 50{\%} sentences of acceptable quality. In addition, many are mislabeled or use nonstandard/ambiguous language codes. We demonstrate that these issues are easy to detect even for non-proficient speakers, and supplement the human audit with automatic analyses. Finally, we recommend techniques to evaluate and improve multilingual corpora and discuss potential risks that come with low-quality data releases.",
}
@inproceedings{ortiz-suarez-etal-2020-monolingual,
    title = "A Monolingual Approach to Contextualized Word Embeddings for Mid-Resource Languages",
    author = "Ortiz Su{\'a}rez, Pedro Javier  and
      Romary, Laurent  and
      Sagot, Benoit",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.156",
    pages = "1703--1714",
    abstract = "We use the multilingual OSCAR corpus, extracted from Common Crawl via language classification, filtering and cleaning, to train monolingual contextualized word embeddings (ELMo) for five mid-resource languages. We then compare the performance of OSCAR-based and Wikipedia-based ELMo embeddings for these languages on the part-of-speech tagging and parsing tasks. We show that, despite the noise in the Common-Crawl-based OSCAR data, embeddings trained on OSCAR perform much better than monolingual embeddings trained on Wikipedia. They actually equal or improve the current state of the art in tagging and parsing for all five languages. In particular, they also improve over multilingual Wikipedia-based contextual embeddings (multilingual BERT), which almost always constitutes the previous state of the art, thereby showing that the benefit of a larger, more diverse corpus surpasses the cross-lingual benefit of multilingual embedding architectures.",
}
@inproceedings{OrtizSuarezSagotRomary2019,
  author    = {Pedro Javier {Ortiz Su{\'a}rez} and Benoit Sagot and Laurent Romary},
  title     = {Asynchronous pipelines for processing huge corpora on medium to low resource infrastructures},
  series = {Proceedings of the Workshop on Challenges in the Management of Large Corpora (CMLC-7) 2019. Cardiff, 22nd July 2019},
  editor    = {Piotr Bański and Adrien Barbaresi and Hanno Biber and Evelyn Breiteneder and Simon Clematide and Marc Kupietz and Harald L{\"u}ngen and Caroline Iliadi},
  publisher = {Leibniz-Institut f{\"u}r Deutsche Sprache},
  address   = {Mannheim},
  doi       = {10.14618/ids-pub-9021},
  url       = {http://nbn-resolving.de/urn:nbn:de:bsz:mh39-90215},
  pages     = {9 -- 16},
  year      = {2019},
  abstract  = {Common Crawl is a considerably large, heterogeneous multilingual corpus comprised of crawled documents from the internet, surpassing 20TB of data and distributed as a set of more than 50 thousand plain text files where each contains many documents written in a wide variety of languages. Even though each document has a metadata block associated to it, this data lacks any information about the language in which each document is written, making it extremely difficult to use Common Crawl for monolingual applications. We propose a general, highly parallel, multithreaded pipeline to clean and classify Common Crawl by language; we specifically design it so that it runs efficiently on medium to low resource infrastructures where I/O speeds are the main constraint. We develop the pipeline so that it can be easily reapplied to any kind of heterogeneous corpus and so that it can be parameterised to a wide range of infrastructures. We also distribute a 6.3TB version of Common Crawl, filtered, classified by language, shuffled at line level in order to avoid copyright issues, and ready to be used for NLP applications.},
  language  = {en}
}
"""

_BASE_DATA_PAT_FORMAT_STR = "{language}_meta/"
_BASE_CHECKSUM_FILE_NAME = "checksum.sha256"


def _languages():
    """Create the sorted dictionary of language codes, and language names.
    Returns:
      The sorted dictionary as an instance of `collections.OrderedDict`.
    """
    langs = {
        "Afrikaans": "af",
        "Albanian": "sq",
        "Amharic": "am",
        "Arabic": "ar",
        "Aragonese": "an",
        "Armenian": "hy",
        "Assamese": "as",
        "Asturian": "ast",
        "Avaric": "av",
        "Azerbaijani": "az",
        "Bangla": "bn",
        "Bashkir": "ba",
        "Basque": "eu",
        "Belarusian": "be",
        "Bihari languages": "bh",
        "Bishnupriya": "bpy",
        "Bosnian": "bs",
        "Breton": "br",
        "Bulgarian": "bg",
        "Burmese": "my",
        "Catalan": "ca",
        "Cebuano": "ceb",
        "Central Kurdish": "ckb",
        "Chechen": "ce",
        "Chinese": "zh",
        "Chuvash": "cv",
        "Cornish": "kw",
        "Croatian": "hr",
        "Czech": "cs",
        "Danish": "da",
        "Divehi": "dv",
        "Dutch": "nl",
        "Eastern Mari": "mhr",
        "Egyptian Arabic": "arz",
        "English": "en",
        "Esperanto": "eo",
        "Estonian": "et",
        "Filipino": "tl",
        "Finnish": "fi",
        "French": "fr",
        "Galician": "gl",
        "Georgian": "ka",
        "German": "de",
        "Goan Konkani": "gom",
        "Greek": "el",
        "Guarani": "gn",
        "Gujarati": "gu",
        "Haitian Creole": "ht",
        "Hebrew": "he",
        "Hindi": "hi",
        "Hungarian": "hu",
        "Icelandic": "is",
        "Ido": "io",
        "Iloko": "ilo",
        "Indonesian": "id",
        "Interlingua": "ia",
        "Interlingue": "ie",
        "Irish": "ga",
        "Italian": "it",
        "Japanese": "ja",
        "Javanese": "jv",
        "Kalmyk": "xal",
        "Kannada": "kn",
        "Karachay-Balkar": "krc",
        "Kazakh": "kk",
        "Khmer": "km",
        "Komi": "kv",
        "Korean": "ko",
        "Kurdish": "ku",
        "Kyrgyz": "ky",
        "Lao": "lo",
        "Latin": "la",
        "Latvian": "lv",
        "Lezghian": "lez",
        "Limburgish": "li",
        "Lithuanian": "lt",
        "Lojban": "jbo",
        "Lombard": "lmo",
        "Low German": "nds",
        "Lower Sorbian": "dsb",
        "Luxembourgish": "lb",
        "Macedonian": "mk",
        "Maithili": "mai",
        "Malagasy": "mg",
        "Malay": "ms",
        "Malayalam": "ml",
        "Maltese": "mt",
        "Marathi": "mr",
        "Mazanderani": "mzn",
        "Minangkabau": "min",
        "Mingrelian": "xmf",
        "Mirandese": "mwl",
        "Mongolian": "mn",
        "Multilingual": "multi",
        "Nahuatl languages": "nah",
        "Nepali": "ne",
        "Newari": "new",
        "Norwegian": "no",
        "Norwegian Nynorsk": "nn",
        "Occitan": "oc",
        "Odia": "or",
        "Ossetic": "os",
        "Pashto": "ps",
        "Persian": "fa",
        "Piedmontese": "pms",
        "Polish": "pl",
        "Portuguese": "pt",
        "Punjabi": "pa",
        "Quechua": "qu",
        "Romanian": "ro",
        "Russia Buriat": "bxr",
        "Russian": "ru",
        "Sakha": "sah",
        "Sanskrit": "sa",
        "Scottish Gaelic": "gd",
        "Serbian": "sr",
        "Serbian (Latin)": "sh",
        "Sindhi": "sd",
        "Sinhala": "si",
        "Slovak": "sk",
        "Slovenian": "sl",
        "Somali": "so",
        "South Azerbaijani": "azb",
        "Spanish": "es",
        "Sundanese": "su",
        "Swahili": "sw",
        "Swedish": "sv",
        "Swiss German": "gsw",
        "Tajik": "tg",
        "Tamil": "ta",
        "Tatar": "tt",
        "Telugu": "te",
        "Thai": "th",
        "Tibetan": "bo",
        "Turkish": "tr",
        "Turkmen": "tk",
        "Ukrainian": "uk",
        "Emiliano-Romagnolo": "x-eml",
        "Upper Sorbian": "hsb",
        "Urdu": "ur",
        "Uyghur": "ug",
        "Uzbek": "uz",
        "Vietnamese": "vi",
        "Volapük": "vo",
        "Walloon": "wa",
        "Waray": "war",
        "Welsh": "cy",
        "Western Frisian": "fy",
        "Western Mari": "mrj",
        "Western Panjabi": "pnb",
        "Wu Chinese": "wuu",
        "Yiddish": "yi",
        "Yoruba": "yo",
    }

    langs = {v: k for k, v in langs.items()}
    return collections.OrderedDict(sorted(langs.items()))


class Oscar2301Config(datasets.BuilderConfig):
    """OSCAR corpus."""

    def __init__(self, language: str, **kwargs):
        """BuilderConfig for OSCAR.
        Args:
            language (str): It has to contain 2-letter or 3-letter coded strings. For example: "se", "hu", "eml"
            **kwargs: Keyword arguments forwarded to super.
        """
        # Validate the language.
        if language not in _languages():
            raise ValueError("Invalid language: %s " % language)

        name = f"{language}"
        description = (
            f"Original {_languages()[language]} OSCAR dataset from January 2023"
        )
        super(Oscar2301Config, self).__init__(
            name=name, description=description, **kwargs
        )

        # Additional attributes
        self.language = language
        self.base_data_path = _BASE_DATA_PAT_FORMAT_STR.format(language=language)


class Oscar2301(datasets.GeneratorBasedBuilder):
    """OSCAR The Open Super-large Crawled Aggregated coRpus."""

    BUILDER_CONFIGS = [
        Oscar2301Config(  # pylint: disable=g-complex-comprehension
            language=language,
            version=datasets.Version("2023.1.0"),
        )
        for language in _languages()
    ]
    BUILDER_CONFIG_CLASS = Oscar2301Config

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("int64"),
                    "text": datasets.Value("string"),
                    "meta": {
                        "warc_headers": {
                            "warc-record-id": datasets.Value("string"),
                            "warc-date": datasets.Value("string"),
                            "content-type": datasets.Value("string"),
                            "content-length": datasets.Value("int32"),
                            "warc-type": datasets.Value("string"),
                            "warc-identified-content-language": datasets.Value(
                                "string"
                            ),
                            "warc-refers-to": datasets.Value("string"),
                            "warc-target-uri": datasets.Value("string"),
                            "warc-block-digest": datasets.Value("string"),
                        },
                        "identification": {
                            "label": datasets.Value("string"),
                            "prob": datasets.Value("float"),
                        },
                        "harmful_pp": datasets.Value("float"),
                        "tlsh": datasets.Value("string"),
                        "quality_warnings": datasets.Sequence(datasets.Value("string")),
                        "categories": datasets.Sequence(datasets.Value("string")),
                        "sentence_identifications": [
                            {
                                "label": datasets.Value("string"),
                                "prob": datasets.Value("float"),
                            }
                        ],
                    },
                }
            ),
            supervised_keys=None,
            homepage=_URL,
            citation=_CITATION,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        checksum_path = self.config.base_data_path + _BASE_CHECKSUM_FILE_NAME.format(
            language=self.config.language
        )
        checksum_file = dl_manager.download(checksum_path)
        with open(checksum_file, encoding="utf-8") as f:
            data_filenames = [line.split()[1] for line in f if line]
            data_urls = [
                self.config.base_data_path + data_filename
                for data_filename in data_filenames
            ]
        doc_files = dl_manager.download(
            [url for url in data_urls if url.endswith(".jsonl.zst")]
        )
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"doc_files": doc_files}
            ),
        ]

    def _generate_examples(self, doc_files):
        """This function returns the examples in the raw (text) form by iterating on all the files."""
        id_ = 0
        for doc_path in doc_files:
            logger.info("generating examples from = %s", doc_path)

            with open(doc_path, "rb") as fh:
                dctx = zstandard.ZstdDecompressor()
                stream_reader = dctx.stream_reader(fh)
                buffered_reader = io.BufferedReader(stream_reader)
                text_stream = io.TextIOWrapper(buffered_reader, encoding="utf-8")
                for line in text_stream:
                    doc = json.loads(line)
                    meta = doc["metadata"]
                    meta["warc_headers"] = doc["warc_headers"]

                    try:
                        meta["warc_headers"]["warc-identified-content-language"]
                    except KeyError:
                        meta["warc_headers"]["warc-identified-content-language"] = None

                    yield id_, {"id": id_, "text": doc["content"], "meta": meta}
                    id_ += 1
