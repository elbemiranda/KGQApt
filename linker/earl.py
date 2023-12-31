import ujson as json
from linker.goldLinker import GoldLinker
from common.container.linkeditem import LinkedItem
from common.container.uri import Uri
from kb.dbpedia import DBpedia
from common.utility.utility import closest_string


class Earl:
    def __init__(self, path="data/LC-QUAD/EARL/output_original.json"):
        self.parser = DBpedia.parse_uri
        self.gold_linker = GoldLinker()
        with open(path, 'r', encoding="utf-8") as data_file:
            self.raw_data = json.load(data_file)
            self.questions = {}
            for item in self.raw_data:
                self.questions[item["question"]] = item

    def __force_gold(self, golden_list, surfaces, items):
        not_found = []
        intersect = []
        uri_list = []
        for i_item in items:
            for i_uri in i_item.uris:
                for g_item in golden_list:
                    if i_uri in g_item.uris:
                        intersect.append(g_item)

        return intersect
    # def __force_gold(self, golden_list, surfaces, items):
    #     not_found = []
    #     for item in golden_list:
    #         idx = closest_string(item.surface_form, surfaces)
    #         if idx != -1:
    #             if item.uris[0] not in items[idx].uris:
    #                 items[idx].uris[len(items[idx].uris) - 1] = item.uris[0]
    #             surfaces.pop(idx)
    #         else:
    #             not_found.append(item)
    #
    #     for item in not_found:
    #         if len(surfaces) > 0:
    #             idx = surfaces.keys()[0]
    #             items[idx].uris[0] = item.uris[0]
    #             surfaces.pop(idx)
    #         else:
    #             items.append(item)
    #
    #     keys = surfaces.keys()
    #     keys.sort(reverse=True)
    #     for idx in keys:
    #         del items[idx]
    #
    #     return items

    def do(self, qapair, force_gold=False, top=50):
        if qapair.question.text in self.questions:
            item = self.questions[qapair.question.text]
            entities = self.__parse(item, "entities", top)
            relations = self.__parse(item, "relations", top)

            if force_gold:
                gold_entities, gold_relations = self.gold_linker.do(qapair)
                entities_surface = {i: item.surface_form for i, item in enumerate(entities)}
                relations_surface = {i: item.surface_form for i, item in enumerate(relations)}

                entities = self.__force_gold(gold_entities, entities_surface, entities)
                relations = self.__force_gold(gold_relations, relations_surface, relations)

            return entities, relations
        else:
            return None, None

    def __parse(self, dataset, name, top):
        #print(f"\n============= {name} =============")
        output = []
        for item in dataset[name]:
            uris = []
            for uri in item["uris"]:
                #print(uri["uri"])
                uris.append(Uri(uri["uri"], self.parser, uri["confidence"]))
            if len(item["surface"])>0:
                start_index, length = item["surface"]
                surface = dataset["question"][start_index: start_index + length]
            else:
                surface = ""
            #print(surface)
            output.append(LinkedItem(surface, uris[:top]))
        return output
