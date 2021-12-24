import numpy as np
from model.utils import pytorch_cos_sim
from data.dataloader import convert_to_tensor, example_model_setting


def main():
    model_name = 'klue/bert-base'
    model_ckpt = '../Checkpoint/KoSimCSE/kosimcse-klue-bert-base.pt'
    model, tokenizer, device = example_model_setting(model_ckpt, model_name)

    # Corpus with example sentences
    corpus = ['한 남자가 음식을 먹는다.',
              '한 남자가 빵 한 조각을 먹는다.',
              '그 여자가 아이를 돌본다.',
              '한 남자가 말을 탄다.',
              '한 여자가 바이올린을 연주한다.',
              '두 남자가 수레를 숲 속으로 밀었다.',
              '한 남자가 담으로 싸인 땅에서 백마를 타고 있다.',
              '원숭이 한 마리가 드럼을 연주한다.',
              '치타 한 마리가 먹이 뒤에서 달리고 있다.']

    inputs_corpus = convert_to_tensor(corpus, tokenizer, device)

    corpus_embeddings = model.encode(inputs_corpus, device)

    # Query sentences:
    queries = ['한 남자가 파스타를 먹는다.',
               '고릴라 의상을 입은 누군가가 드럼을 연주하고 있다.',
               '치타가 들판을 가로 질러 먹이를 쫓는다.']

    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = 5
    for query in queries:
        query_embedding = model.encode(convert_to_tensor([query], tokenizer, device), device)
        
        cos_scores = pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        cos_scores = cos_scores.cpu().detach().numpy()

        top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nTop 5 most similar sentences in corpus:")

        for idx in top_results[0:top_k]:
            print(corpus[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))


if __name__ == '__main__':
    main()
