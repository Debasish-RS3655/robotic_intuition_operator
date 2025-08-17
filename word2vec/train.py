import gensim
import gensim.downloader as api
from gensim.models import Word2Vec

def main():
    print("⏳ Loading text8 corpus...")
    dataset = api.load("text8")  # this is a stream of tokens
    data = [list(doc) for doc in dataset]  # convert to list-of-lists

    print("✅ Corpus loaded. Training Word2Vec...")
    model = Word2Vec(
        sentences=data,
        vector_size=100,   # you can lower to 50 for even smaller file
        window=5,
        min_count=5,
        workers=4
    )

    # Save gensim model
    model.save("text8.model")
    print("💾 Saved gensim model -> text8.model")

    # Save in classic word2vec binary format
    model.wv.save_word2vec_format("text8.bin", binary=True)
    print("💾 Saved word2vec binary -> text8.bin")

    print("🎉 Done! You can now load text8.bin in Node.js")

if __name__ == "__main__":
    main()
