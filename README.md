# kmeans
KMeans algorithm implemented in Typescript. This is part of the tutorial which you can find
[here](https://medium.com/p/be42930fc1c0)


## Usage

You'll need to supply k and dataset to the algorithm

```typescript
import { KMeans } from './kmeans'

const dataset = [
  [233, 55, 33], // redish
  [255, 29, 199],// redish
  [219, 39, 10], // redish
  [199, 29, 39], // redish
  [52, 198, 33], // greenish
  [39, 209, 22], // greenish
  [0, 255, 29],  // greenish
  [11, 20, 202], // blueish
  [89, 22, 185], // blueish
  [54, 20, 200], // blueish
]

const k = 3

const kmeans = new KMeans(k, dataset)
kmeans.fit()
console.log(kmeans.assignments) // Expects `number[][][]` representing
// clusters arround red, green and blue
```
