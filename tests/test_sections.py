"""Section labels and the second-hand flag. All text is synthetic; each case reproduces one extraction pattern."""
import pytest
from src import ingest
from src.ingest import body_headings, is_secondhand, label_sections, split_sections

ROWS = "12 3.4 0.96 0.02\n" * 8
FILLER = "Plain sentences about nothing in particular fill the space here.\n"


def labels(text):
    return [label for label, _ in label_sections(text)]


def section(text, label):
    return "".join(t for lab, t in label_sections(text) if lab == label)


def test_small_caps_and_letter_spaced_headings():
    text = ("AbsTRACT\nObjective of the work.\nIn\nTROD\nu\nCTIO\nn\nOpening text.\nMEThODs\nWe trained men.\n"
            "REsulT s\nGains rose.\nDIsCussIOn\nWhy it happened.\nCOnClusIOn\nDone.\n")
    assert labels(text) == ["abstract", "introduction", "methods", "results", "discussion"]
    assert "Opening text" in section(text, "introduction") and "Gains rose" in section(text, "results")


def test_body_headings_early_in_a_long_paper_are_not_taken_for_a_structured_abstract():
    text = ("Abstract\nShort abstract.\nKeywords: strength, muscle\nBackground\nOpening text.\nMethods\nWe searched.\n"
            "Results\nWe found 18.\nDiscussion\n" + FILLER * 300)  # all headings sit in the first 5%
    assert labels(text) == ["abstract", "introduction", "methods", "results", "discussion"]


def test_numbered_background_ends_the_abstract():
    text = "Abstract\nBackground\nAims.\nResults\nGains.\n1  Background\nOpening text.\n2  Methods\nWe searched.\n3  Results\nFound.\n"
    assert labels(text) == ["abstract", "introduction", "methods", "results"]
    assert "Gains" in section(text, "abstract") and "Opening text" in section(text, "introduction")


def test_introduction_without_a_heading_starts_after_the_keywords():
    text = "ABSTRACT\nWe pooled trials. Key Words: STRENGTH\nTraining is popular.\nMETHODS\nWe searched.\nRESULTS\nFound.\n"
    assert labels(text) == ["abstract", "introduction", "methods", "results"]
    assert "Training is popular" in section(text, "introduction")


def test_abstract_without_keywords_or_introduction_ends_at_the_first_heading_after_the_maximum():
    text = "Summary\n" + FILLER * 80 + "METHODS\nWe searched.\n" + FILLER * 200
    assert labels(text) == ["abstract", "methods"]


@pytest.mark.parametrize("heading", ["MATERIAL AND METHODS", "Materials and Methods", "Methodology", "87 MATERIAL AND METHODS"])
def test_methods_heading_variants(heading):
    assert labels(f"INTRODUCTION\nOpening.\n{heading}\nWe trained.\nRESULTS\nFound.\n") == ["introduction", "methods", "results"]


def test_narrative_review_topic_sections_end_the_introduction():
    text = ("Abstract\nAims.\n1 Introduction\nOpening text.\n1 School of Sport, \n2 Department of Science and Health\n"
            "3 Institute for Exercise\n" + FILLER * 8 + "2  Training Frequency\nTwice a week.\n" + FILLER * 8 +
            "3  Training Volume\n3.1  Sets\nOne set.\n" + FILLER * 8 + "4  Conclusion and Recommendations\nDo something.\n")
    assert labels(text) == ["abstract", "introduction", "discussion"]
    assert "Department of Science" in section(text, "introduction") and "Twice a week" in section(text, "discussion")


def test_topic_section_before_methods_stays_in_the_introduction():
    text = ("1 Introduction\nOpening.\n" + FILLER * 8 + "2 Conceptual Framework\nTheory.\n" + FILLER * 8 +
            "3 Methods\nWe searched.\n4 Results\nFound.\n")
    assert labels(text) == ["introduction", "methods", "results"]


def test_table_column_named_like_a_section_is_ignored_when_the_real_one_is_numbered():
    text = ("1 Introduction\nOpening.\n2 Methods\nWe searched.\nConclusion\nSignificant\nResults\nyes\n3 Results\nFound.\n"
            "4 Discussion\nWhy.\n5 Conclusion\nDone.\n")
    assert [h.number for h in body_headings(text)] == ["1", "2", "3", "4"]


def test_wrapped_lowercase_line_is_not_a_heading():
    assert labels("Introduction\nEarlier work compared\nresults\nacross groups.\nDiscussion\nWhy.\n") == ["introduction", "discussion"]


def test_tables_get_their_own_label_until_the_page_ends_or_prose_resumes():
    prose = "this sentence runs on with plenty of ordinary lowercase words in it\n"
    text = ("Discussion\n" + prose * 8 + "\f\nTable 3 (continued)\nDevice A 0.96 0.02\nDevice B 0.91 0.03\n" + ROWS + "\f\n" + prose * 8 +
            "Table 4 Summary of devices\nDevice C 0.88 0.05\nDevice D 0.85 0.06\n" + ROWS + prose * 8)
    assert labels(text) == ["discussion", "table", "discussion", "table", "discussion"]
    assert "Device A" not in section(text, "discussion") and prose not in section(text, "table")


@pytest.mark.parametrize("mention", ["Table 2 shows that scores rose.", "Table 2.", "Table 2). In ten studies"])
def test_mentions_of_a_table_are_not_captions(mention):
    assert labels(f"Results\nScores were high.\n{mention}\nMore text follows here.\n") == ["results"]


def test_heading_inside_a_table_is_not_a_section():
    text = "Results\nFound.\n\f\nTable 1 Included studies\nStudy\nDiscussion\n" + ROWS + "\f\nMore findings.\n"
    assert labels(text) == ["results", "table", "results"]


def test_text_after_the_references_is_table():
    text = "Introduction\nAims.\nDiscussion\n" + FILLER * 20 + "References\n1. Smith J.\nFigure 1. Strength changes\n" + ROWS
    kept = split_sections(text)[1]
    assert label_sections(kept)[-1][0] == "table" and "Smith J" not in kept
    assert "Figure 1" in section(kept, "table") and "Figure 1" not in section(kept, "discussion")


def test_labels_lose_no_text():
    text = "Title\nAbstract\nAims.\n1 Introduction\nA.\n\f\nTable 1 Data\n" + ROWS + "\f\n2 Methods\nB.\n"
    assert "".join(t for _, t in label_sections(text)).split() == text.split()


@pytest.mark.parametrize("text", [
    "Doe et al. (6) reported that ten weeks of training raised muscle size. Roe and Poe (2011) found the same in women.",
    "In this context, Doe et al. [66] compared one, three and five sets. Five sets were best [67, 68].",
    "Heavy loads recruit more units [12–14]. Light loads to failure may do the same [15].",
    "Protein has been widely studied1–5 and reviews disagree6,7 on its effect in trained adults.",
    "Gains were larger in trained men (12). Others saw no difference (13, 14).",
])
def test_secondhand_is_true_for_text_about_other_studies(text):
    assert is_secondhand(text, "discussion") is True


@pytest.mark.parametrize("text", [
    "We found that strength rose by 10% in both groups. Muscle size did not change.",
    "Our results agree with Doe et al. [12], who saw similar gains. We also found no effect of sex.",
    "The present study showed larger gains with more sets, in line with earlier work [3, 4].",
    "The effect was small (ES = 0.14; 95% CI 0.01 to 0.29). Groups did not differ (p = 0.06). One study agreed [9].",
    "Participants lifted 3 sets of 8-12 repetitions (80% 1RM) twice a week for 10 weeks.",
    "",
])
def test_secondhand_is_false_for_the_papers_own_work(text):
    assert is_secondhand(text, "results") is False


def test_secondhand_ultrasound_abbreviation_is_not_own_work_and_tables_and_methods_are_exempt():
    text = "Doe et al. [6] measured size with US imaging and reported gains."
    assert is_secondhand(text, "discussion") and not is_secondhand(text, "table") and not is_secondhand(text, "methods")


def test_vector_store_saves_the_secondhand_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "embed", lambda texts: [[1.0, 0.5, 0.2] for _ in texts])
    store = ingest.VectorStore(path=tmp_path)
    store.add("a_2020", ["We found gains.", "Doe et al. [6] reported gains."], {"title": "T"}, ["results", "discussion"])
    assert {h["chunk_index"]: h["secondhand"] for h in store.query("q")} == {0: False, 1: True}
