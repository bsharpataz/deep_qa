import codecs
import nltk


class Question:
    def __init__(self, question, answers, label, index):
        self.question = question
        self.answers = answers
        self.label = label
        self.index = index
        self.q_tokens = self.get_q_tokens()

    def display(self):
        print("Q:", self.question)
        print("Answers:", self.answers)
        print("Label:", self.label)
        print("Index:", self.index)

    def get_q_tokens(self):
        tokens = []
        proc_sents = nltk.sent_tokenize(self.question)
        proc_words = [nltk.word_tokenize(s) for s in proc_sents]
        for word_list in proc_words:
            for word in word_list:
                if word.isalnum():
                    tokens.append(word.lower())
        return tokens

    ## todo: Implement a method which gets only the noun/verb/(JJ and RB?) tokens


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


def question_overlap(question: Question, tuples: list, tuple_portion: str):
    qText = set(question.q_tokens)
    tupleText = [get_relevant_tuple_portion(t, tuple_portion) for t in tuples]
    # print("qText:", qText)
    # print("tupleText:", tupleText)
    for t in tupleText:
        # print(t)
        overlap = set(t.split(" ")).intersection(qText)
        if len(overlap) > 0:
            #print("Overlap:", overlap)
            return True
    return False


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

    question_overlap(indexed_questions[qIdx], indexed_tuples[qIdx], "S")

    print("")
    print("------------------------------------")
    print("          Overlap Analysis")
    print("------------------------------------")
    print(len(indexed_questions))
    print(len(indexed_tuples))

    tuple_portion = "O"
    for i in range(0, len(indexed_questions)):
        curr_question = indexed_questions[i]
        curr_tuples = indexed_tuples.get(i, [])
        # if len(indexed_tuples[i]) < 2:
        #     print("ISSUE: indexed_tuples[{0}]: {1}".format(i, indexed_tuples[i]))
        # curr_tuples = indexed_tuples[i][1]
        print("Question {0} has {2} overlap with at least one tuple: {1}".format(i, question_overlap(curr_question, curr_tuples, tuple_portion), tuple_portion))


main()