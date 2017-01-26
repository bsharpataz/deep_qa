from __future__ import division
import codecs
import nltk
from nltk.corpus import stopwords

stops = set(stopwords.words('english'))

class Question:
    def __init__(self, question, answers, label, index):
        self.question = question
        self.answers = answers
        self.label = int(label)
        self.index = index
        self.q_tokens = self.get_tokens(self.question)
        self.a_tokens = [self.get_tokens(a) for a in self.answers]

    def display(self):
        print("Q:", self.question)
        print("Answers:", self.answers)
        print("Label:", self.label)
        print("Index:", self.index)

    def get_tokens(self, text):
        tokens = []
        proc_sents = nltk.sent_tokenize(text)
        proc_words = [nltk.pos_tag(nltk.word_tokenize(s)) for s in proc_sents]
        for word_list in proc_words:
            for (word, tag) in word_list:
                if self.is_nounverb_tag(tag) and word not in stops:
                    tokens.append(word.lower())
        return tokens

    def is_nounverb_tag(self, tag):
        return tag.startswith("NN") | tag.startswith("VB")

    # todo: filter out with nltk stopwords?


def read_questions_from_file(filename: str):
    questions_out = dict()
    lines = [x.strip() for x in codecs.open(filename, "r", "utf-8").readlines()]
    for line in lines:
        question, answer_options, label, index = question_read_from_line(line)
        questions_out[index] = Question(question, answer_options, label, index)
    return questions_out


def read_tuples_from_file(filename: str):
    tuples_out = dict()
    tuples_hist = dict()
    for line in codecs.open(filename, "r", "utf-8"):
        fields = line.strip().split("\t")
        index = int(fields[0])
        curr_tuples = []
        for sequence in fields[1:]:
            tuples = tuple_read_from_line(sequence, None)
            curr_tuples.append(tuples)
        tuples_hist[len(curr_tuples)] = tuples_hist.get(len(curr_tuples), 0) + 1
        tuples_out[index] = curr_tuples

    print("Histogram of num tuples:", tuples_hist)
    hist_sum = 0
    for k,v in tuples_hist.items():
        print("{0},{1}".format(k,v))
        hist_sum += v
    running_sum = 0
    print("running sum:")
    for k in tuples_hist.keys():
        tuples_hist[k] /= hist_sum
        running_sum += tuples_hist[k]
        print("{0},{1}".format(k, running_sum))
    print("(Normalized) Histogram of num tuples:", tuples_hist)

    return tuples_out


# TupleInstance.read_from_line
def tuple_read_from_line(line: str, default_label: bool=True):
    """
    Reads a TupleInstances from a line.  The format has one of four options:

    (1) [subject]###[predicate]###[object1]...
    (2) [sentence index][tab][subject]###[predicate]###[object1]...
    (3) [subject]###[predicate]###[object1]...[tab][label]
    (4) [sentence index][tab][subject]###[predicate]###[object1]...[tab][label]

    Objects are optional, and can vary in number. This makes the acceptable number of slots per
    tuple, 2 or more.
    """
    fields = line.split("\t")
    if len(fields) == 1:
        # Case 1
        tuple_string = fields[0]
        index = None
        label = default_label
    elif len(fields) == 2:
        if fields[0].isdigit():
            # Case 2
            index = int(fields[0])
            tuple_string = fields[1]
            label = default_label
        else:
            # Case 3
            tuple_string = fields[0]
            index = None
            label = fields[2].strip() == "1"
    else:
        # Case 4
        index = int(fields[0])
        tuple_string = fields[1]
        label = fields[2].strip() == "1"
    tuple_fields = tuple_string.split('###')
    if len(tuple_fields) < 2:
        raise RuntimeError("Unexpected number of fields in tuple: " + tuple_string)
    return tuple_fields


# QuestionAnswerInstance.read_from_line
def question_read_from_line(line: str):
    """
    Reads a QuestionAnswerInstance object from a line.  The format has two options:

    (1) [question][tab][answer_options][tab][correct_answer]
    (2) [instance index][tab][question][tab][answer_options][tab][correct_answer]

    The `answer_options` column is assumed formatted as: [option]###[option]###[option]...
    That is, we split on three hashes ("###").

    default_label is ignored, but we keep the argument to match the interface.
    """
    fields = line.split("\t")

    if len(fields) == 3:
        question, answers, label_string = fields
        index = None
    elif len(fields) == 4:
        if fields[0].isdecimal():
            index_string, question, answers, label_string = fields
            index = int(index_string)
        else:
            raise RuntimeError("Unrecognized line format: " + line)
    else:
        raise RuntimeError("Unrecognized line format: " + line)
    answer_options = answers.split("###")
    label = int(label_string)
    return question, answer_options, label, index


def get_relevant_tuple_portion(t: list, tuple_portion: str):
    if tuple_portion == "SV":
        return " ".join(t[0:1])
    if tuple_portion == "O":
        if len(t) < 3:
            return ""
        else:
            return " ".join(t[2:])
    if tuple_portion == "S":
        return t[0]
    if tuple_portion == "VO":
        return " ".join(t[1:])
    else:
        raise NotImplementedError("ERROR: Invalid tuple_portion selected ({0})".format(tuple_portion))


def flatten(l):
    if len(l) == 1:
        return l
    else:
        return [item for sublist in l for item in sublist]


def getQATokens(question: Question, mode: str):
    mode_options = {'question': question.q_tokens,  # list of strings
                    'correct_answer': question.a_tokens[question.label], # list of strings
                    'incorrect_answers': flatten(question.a_tokens[0:question.label] + question.a_tokens[question.label:len(question.a_tokens)]),
                    'all_answers': flatten(question.a_tokens[:])}
    return mode_options[mode]


def check_overlap(tokens: set, tuples: list, tuple_portion: str):
    tuple_texts = [get_relevant_tuple_portion(t, tuple_portion) for t in tuples]
    for t in tuple_texts:
        # print(t)
        overlap = set(t.split(" ")).intersection(tokens)
        if len(overlap) > 0:
            #print("Overlap:", overlap)
            return 1
    return 0

def question_overlap(question: Question, mode: str, tuples: list):

    qa_tokens = set(getQATokens(question, mode))

    # Stats holders
    in_s = check_overlap(qa_tokens, tuples, "S")
    in_sv = check_overlap(qa_tokens, tuples, "SV")
    in_vo = check_overlap(qa_tokens, tuples, "VO")
    in_o = check_overlap(qa_tokens, tuples, "O")

    return [in_s, in_sv, in_vo, in_o]


def format_perc(percent: float):
    return "%.1f" % (percent*100)


def main():
    # Data
    questions_file = "/Users/rebeccas/data/questions.tsv"
    tuples_file = "/Users/rebeccas/data/questions_background_as_tuples.tsv"

    # Load the questions and tuples
    indexed_questions = read_questions_from_file(questions_file)
    indexed_tuples = read_tuples_from_file(tuples_file)

    qIdx = 2
    #print("question {0}: {1}".format(qIdx, indexedQuestions[qIdx]))
    indexed_questions[qIdx].display()
    print("Tuples:")
    for t in indexed_tuples[qIdx]:
        print("   ", t)

    question_overlap(indexed_questions[qIdx], "question", indexed_tuples[qIdx])

    print("")
    print("------------------------------------")
    print("          Overlap Analysis")
    print("------------------------------------")
    print("num questions:", len(indexed_questions))
    print("num questions with tuples:", len(indexed_tuples))
    print("")

    # Stats holders
    num_q = len(indexed_questions)
    num_q_svOverlap = 0
    num_q_voOverlap = 0
    num_ca_oOverlap = 0
    num_ca_sOverlap = 0
    num_aligned_qSV_caO = 0
    num_aligned_qVO_caS = 0
    num_ia_oOverlap = 0
    num_ia_sOverlap = 0
    num_aligned_qSV_iaO = 0
    num_aligned_qVO_iaS = 0
    num_aligned_strict_qSV_caO = 0
    num_aligned_strict_qVO_caS = 0


    for i in range(0, len(indexed_questions)):
        curr_question = indexed_questions[i]
        curr_tuples = indexed_tuples.get(i, [])
        # Question
        q_in_s, q_in_sv, q_in_vo, q_in_o = question_overlap(curr_question, "question", curr_tuples)
        num_q_svOverlap += q_in_sv
        num_q_voOverlap += q_in_vo
        # Correct answer only
        ca_in_s, ca_in_sv, ca_in_vo, ca_in_o = question_overlap(curr_question, "correct_answer", curr_tuples)
        num_ca_oOverlap += ca_in_o
        num_ca_sOverlap += ca_in_s
        # Incorrect answers only
        ia_in_s, ia_in_sv, ia_in_vo, ia_in_o = question_overlap(curr_question, "incorrect_answers", curr_tuples)
        num_ia_oOverlap += ia_in_o
        num_ia_sOverlap += ia_in_s
        # Aligned with structured intuition:
        # Holds for correct answer (good)
        if q_in_sv and ca_in_o:
            num_aligned_qSV_caO += 1
        if q_in_vo and ca_in_s:
            num_aligned_qVO_caS += 1
        # Holds for incorrect answers (badish...)
        if q_in_sv and ia_in_o:
            num_aligned_qSV_iaO += 1
        if q_in_vo and ia_in_s:
            num_aligned_qVO_iaS += 1
        # Strict - holds for correct and not for incorrect (awesome!)
        if q_in_sv and ca_in_o and not ia_in_o:
            num_aligned_strict_qSV_caO += 1
        if q_in_vo and ca_in_s and not ia_in_s:
            num_aligned_strict_qVO_caS += 1



        #print("Question {0} has the following overlap with at least one tuple: {1}".format(i, [q_in_s, q_in_sv, q_in_vo, q_in_o]))

    print("Percent of questions with question SV lexical overlap: ", format_perc(num_q_svOverlap/num_q))
    print("Percent of questions with question VO overlap: ", format_perc(num_q_voOverlap / num_q))
    print("Percent of questions whose correct answers (ca) have O overlap: ", format_perc(num_ca_oOverlap / num_q))
    print("Percent of questions whose correct answers (ca)  have S overlap: ", format_perc(num_ca_sOverlap / num_q))

    print("--------------------------------")
    print("Percent of questions with ALIGNED q-SV, ca-O overlap: ", format_perc(num_aligned_qSV_caO / num_q))
    print("Percent of questions with ALIGNED q-VO, ca-S overlap: ", format_perc(num_aligned_qVO_caS / num_q))
    # todo: are these disjoint?

    print("--------------------------------")
    print("Percent of questions with ALIGNED q-SV, incorrect answer (ia) O overlap: ", format_perc(num_aligned_qSV_iaO / num_q))
    print("Percent of questions with ALIGNED q-VO, ia-S overlap: ", format_perc(num_aligned_qVO_iaS / num_q))

    print("--------------------------------")
    print("Percent STRICT ALIGNED q-SV, ca-O overlap, not ia-O overlap: ", format_perc(num_aligned_strict_qSV_caO / num_q))
    print("Percent STRICT ALIGNED q-VO, ca-S overlap, not ia-S overlap: ", format_perc(num_aligned_strict_qVO_caS / num_q))



# todo: % questions overlap with S, SV, VO, O (and same for answers, and then for correct answer/incorrect answers only?)
# todo: break down that percent by 100%, 50%, 25% of tuples? or perhaps by 1-5 k tuples?
# todo: check the "forced" alignment -- i.e. Q with SV and A with O and alternatively Q with VO and A with S (break down for correct and incorrect?)


main()