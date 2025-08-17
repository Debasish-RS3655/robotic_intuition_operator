const word2vec = require('word2vec');

// Load the pre-trained word2vec model and prints the vector for the word 'happy'
word2vec.loadModel('text8.bin', (err, model) => {
  if (err) throw err;
  console.log("Model loaded!");
  console.log("Vector for 'happy':", model.getVector('happy'));
});
