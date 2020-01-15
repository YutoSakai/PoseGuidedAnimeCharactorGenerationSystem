import CaboCha
import json
import time

start = time.time()
c = CaboCha.Parser()

count = 0

sentence = "太郎はこの本を二郎を見た女性に渡した。"
f = open("/home/yuto/PycharmProjects/CGS/newspaper/paper_data/texts.json", 'r')
sentences_json = json.load(f)
for sentence_json in sentences_json.values():
    # print(c.parseToString(sentence_json))
    count += len(sentence_json)

    tree = c.parse(sentence_json)

    # print(tree.toString(CaboCha.FORMAT_TREE))
    print(tree.toString(CaboCha.FORMAT_LATTICE))

elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
print("文字数:" + str(count))