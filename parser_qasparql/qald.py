import json, re
from common.container.qapair import QApair
from common.container.uri import Uri
from common.container.answerrow import AnswerRow
from common.container.answer import Answer
from kb.dbpedia import DBpedia
from parser_qasparql.answerparser import AnswerParser
from xml.dom import minidom
import sys


class Qald:
    qald_7 = "./data/QALD/qald-7-train-multilingual.json"
    qaldtest_7 = "./data/QALD/qald-7-test-multilingual.json"

    def __init__(self, path):
        self.raw_data = []
        self.qapairs = []
        self.path = path
        self.parser = QaldParser()

    def load(self, path=None):
        if path is None:
            path = self.path
        if path.endswith("json"):
            with open(path, encoding='utf-8') as data_file:
                self.raw_data = json.load(data_file)
        elif path.endswith("xml"):
            with open(path) as data_file:
                self.raw_data = minidom.parse(data_file).documentElement

    def extend(self, path):
        self.load(path)
        self.parse()

    def parse(self):
        if self.path.endswith("json"):
            self.parse_json()
        elif self.path.endswith("xml"):
            self.parse_xml()

    def parse_json(self):
        parser = QaldParser()
        for raw_row in self.raw_data["questions"]:
            question = ""
            query = ""
            if "question" in raw_row:
                question = raw_row["question"]
            elif "body" in raw_row:
                # QALD-5 format
                question = raw_row["body"]
            if "query" in raw_row:
                if isinstance(raw_row["query"], dict):
                    if "sparql" in raw_row["query"]:
                        query = raw_row["query"]["sparql"]
                    else:
                        query = ""
                else:
                    query = raw_row["query"]
            self.qapairs.append(QApair(question, raw_row["answers"], query, raw_row, raw_row["id"], parser))

    def parse_xml(self):
        parser = QaldParser()
        data_set = self.raw_data

        raw_rows = data_set.getElementsByTagName("question")
        for raw_row in raw_rows:
            question = []
            answers = []
            query = ""
            question_id = raw_row.getAttribute("id")

            if raw_row.getElementsByTagName("query"):
                query = raw_row.getElementsByTagName("query")[0].childNodes[0].data
            elif raw_row.getElementsByTagName("pseudoquery"):
                query = raw_row.getElementsByTagName("pseudoquery")[0].childNodes[0].data
            query = query.replace("\n"," ")
            query = re.sub(r" {2,}","",query)

            questions_text = raw_row.getElementsByTagName('string')
            questions_keyword = raw_row.getElementsByTagName('keywords')
            for i in range(0,len(questions_text)):
                lang = questions_text[i].getAttribute("lang")
                string = questions_text[i].childNodes
                if string:
                    string = string[0].data
                else:
                    string = ""
                if questions_keyword:
                    keyword = questions_keyword[i].childNodes
                    if keyword:
                        keyword = keyword[0].data
                    else:
                        keyword = ""
                else:
                    keyword = ""
                question.append({u"language":lang, u"string":string, u"keywords": keyword})

            answer_row = raw_row.getElementsByTagName("answers")[0]
            answers_list = answer_row.getElementsByTagName("answer")
            for a in range(0,len(answers_list)):
                answers.append({u"string": u"{}".format(answers_list[a].childNodes[0].data) })
            self.qapairs.append(QApair(question, answers, query, raw_row, question_id, parser))

    def print_pairs(self, n=-1):
        for item in self.qapairs[0:n]:
            print(item)
            print("")


class QaldParser(AnswerParser):
    def __init__(self):
        super(QaldParser, self).__init__(DBpedia())

    def parse_question(self, raw_question):
        # print "AA", raw_question
        for q in raw_question:
            if q["language"] == "pt_BR":
                return q["string"]

    def parse_sparql(self, raw_query):
        if "sparql" in raw_query:
            raw_query = raw_query["sparql"]
        elif isinstance(raw_query, str) and "where" in raw_query.lower():
            pass
        else:
            raw_query = ""
        if "PREFIX " in raw_query:
            # QALD-5 bug!
            raw_query = raw_query.replace("htp:/w.", "http://www.")
            raw_query = raw_query.replace("htp:/dbpedia.", "http://dbpedia.")

            for item in re.findall("PREFIX [^:]*: <[^>]*>", raw_query):
                prefix = item[7:item.find(" ", 9)]
                uri = item[item.find("<"):-1]
                raw_query = raw_query.replace(prefix, uri)
            idx = raw_query.find("WHERE")
            idx2 = raw_query[:idx - 1].rfind(">")
            raw_query = raw_query[idx2 + 1:]
            for uri in set(re.findall('<[^ ]+', raw_query)):
                if uri[-1] != '>':
                    raw_query = raw_query.replace(uri, uri + ">")

        uris = [Uri(raw_uri, self.kb.parse_uri) for raw_uri in re.findall('<[^>]*>', raw_query)]
        supported = not any(substring in raw_query for substring in ["UNION", "FILTER", "OFFSET", "HAVING", "LIMIT"])
        return raw_query, supported, uris

    def parse_answerset(self, raw_answers):
        if len(raw_answers) == 0:
            return []
        elif len(raw_answers) == 1:
            return self.parse_queryresult(raw_answers[0])
        else:
            result = []
            for item in raw_answers:
                result.append(
                    AnswerRow(item["string"],
                              lambda x: [Answer("uri", x, lambda t, y: ("uri", Uri(y, self.kb.parse_uri)))]))

            return result

    def parse_answerrow(self, raw_answerrow):
        answers = [Answer(raw_answerrow["AnswerType"], raw_answerrow, self.parse_answer)]
        return answers

    def parse_answer(self, answer_type, raw_answer):
        if answer_type == "boolean":
            return answer_type, str(raw_answer)
        else:
            if not answer_type in raw_answer:
                answer_type = "\"{}\"".format(answer_type)
            return raw_answer[answer_type]["type"], Uri(raw_answer[answer_type]["value"], self.kb.parse_uri)
