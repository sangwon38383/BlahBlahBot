import gensim  
import argparse
import json 
from konlpy.tag import Kkma 

#대충 만들어 놓을테니 크롤링 된 결과 보고 조금만 바꿔서 씁시다.
#단어list 가 들어오면 임베딩을 시킵니다.

def parser_add_argument(parser):
    parser.add_argument("--crawled_file_1p")
    parser.add_argument("--crawled_file_2p")
    parser.add_argument("--save_path")
    return parser

#json 파일에 {"id": "사람 이름", "text": "크롤링 결과들"} 이렇게 들어옮을 전제로 만들었습니다. 
#json 파일을 읽습니다.

def jsonreader(json):
    output = []
        text = json.loads(json)
        text = text["sentences"]
        for i in text:
            output.append(text)
    return output

#문장에서 명사를 추출합니다.

kkma = Kkma()

def wordsampler(sent):
    sent = kkma.nouns(sent)
    return sent 

#임베딩을 합니다.

model = gensim.models.Word2Vec.load('./ko.bin')

def embedder(words):
    
    embs = []
    for i in words:    
        emb = model.wv[i]
        embs.append(emb)

    return embs

#임베딩을 시켰으니 코사인 거리를 재고, 공통 관심사를 뽑습니다. 
#일단 2인 모드로 만들어 두겠습니다. 

def matcher(as, bs):
    dist_list = []
    for a in as:
        for b in bs:
            similarity = model.similarity(w1=a, w2=b)
            pair = [a, b, similarity]
            dist_list.append(pair)

    dist_list.sort(key=2, reverse=True)
    dist_close_pair = dist_list[:9]
    themes_and_dists = []
    for d in dist_close_pair:
        themes_and_dists.append(d[0],d[2])
        
    return themes_and_dists

#matcher는 주제들의 리스트(10가지)을 출력합니다. 
 
def main():
    parser = argparse.ArgumentParser()
    parser = parser_add_argument(parser)
    args = parser.parse_args()

    sents_1 = jsonreader(args.crawled_file_1p)
    sents_2 = jsonreader(args.crawled_file_2p)

    1p_embs = []
    2p_embs = []

    for sent in sents_1:
        sent = wordsampler(sent)
        embs = embedder(sent)
        1p_embs.append(embs)

    for sent in sents_2:
        sent = wordsampler(sent)
        embs = embedder(sent)
        2p_embs.append(embs)

    themes_dists = matcher(1p_embs, 2p_embs)
    
    out_file = {"themes":[]}
    for t, d in themes_dists:
        out_file["themes"].append({"title":"오늘은 %s에 대해 얘기해볼까요?"%t, "similarity":d})
        out_file["themes"].append({"title":"%s에 대해 관심이 많으신가요?"%t, "similarity":d})
        out_file["themes"].append({"title":"안녕하세요, 혹시 두분 다 %s에 관심이 많으실 것 같은데, 이거와 관련해 얘기해볼까요?"%t, "similarity":d}) 
 
#json으로 저장합니다.
   
    with open(args.save_path, 'w') as outfile:
    json.dummp(out_file, outfile)

 
if __name__ == "__main__":
    main()
             
        
    


