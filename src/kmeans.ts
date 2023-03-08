export function curry(func: Function) {
  return function curried(...args: any[]) {
    if (args.length >= func.length) {
      return func.apply(null, args)
    } else {
      return function (...args2: any[]) {
        return curried.apply(null, args.concat(args2))
      }
    }
  }
}

export function minikowski(p: number, x: number[], y: number[]) {
  const sum = x.reduce(
    (acc, curr, index) => acc + Math.abs(curr - y[index]) ** p,
    0
  )
  return sum ** (1 / p)
}

const curriedMinikowski = curry(minikowski)

/**
 * Compute the eucledian distance
 */
export const eucledian: (x: number[], y: number[]) => number =
  curriedMinikowski(2)

/**
 * Compute the manhattan distance
 */
export const manhattan: (x: number[], y: number[]) => number =
  curriedMinikowski(1)

/**
 * Configuration object for the KMeans
 */
export interface KMeansConfig {
  initializer: 'RANDOM' | 'GENETIC'
  maxIterations: number
}

/**
 * Cluster dataset into K groups centered around mean points.
 */
export class KMeans {
  private _epochs: number
  protected dimension: number
  protected maxValues: number[]
  protected minValues: number[]

  get epochs() {
    return this._epochs
  }

  /**
   * Create new KMeans algorithm instance.
   * @param k Number of clusters
   */
  constructor(
    public readonly k: number,
    public readonly dataset: number[][],
    public centroids: number[][] = []
  ) {
    this._epochs = 0
    this.dimension = dataset[0].length
    this.maxValues = Array(this.dimension).fill(Number.MIN_SAFE_INTEGER)
    this.minValues = Array(this.dimension).fill(Number.MAX_SAFE_INTEGER)
  }

  // Randomly select k data points as initial centroids
  protected selectRandomCentroids() {
    for (const point of this.dataset) {
      for (let i = 0; i < this.dimension; i++) {
        this.minValues[i] = Math.min(this.minValues[i], point[i])
        this.maxValues[i] = Math.max(this.maxValues[i], point[i])
      }
    }

    const centroids: number[][] = Array(this.k).fill(
      Array(this.dimension).fill(NaN)
    )

    for (let i = 0; i < this.k; i++) {
      const centroid: number[] = []
      for (let j = 0; j < this.dimension; j++) {
        centroid.push(
          Math.random() * (this.maxValues[j] - this.minValues[j]) +
            this.minValues[j]
        )
      }
      centroids.push(centroid)
    }
    return centroids.filter((centroid) => centroid.every((el) => !isNaN(el)))
  }

  // Compute distance
  protected distance(fromPoint: number[], toPoint: number[]) {
    return eucledian(fromPoint, toPoint)
  }

  // Assign every point in the dataset to the nearest centroid
  private assignToNearbyCentroids() {
    const clusters = new Map<number, number[]>()
    for (let i = 0; i < this.dataset.length; i++) {
      let closestCentroidIdx = -1
      let closestDistance = Number.MAX_SAFE_INTEGER

      for (let j = 0; j < this.k; j++) {
        if (this.centroids[j] == null || this.dataset[i] == null) continue

        const distance = eucledian(this.dataset[i], this.centroids[j])
        if (distance < closestDistance) {
          closestCentroidIdx = j
          closestDistance = distance
        }
      }

      if (!clusters.has(closestCentroidIdx)) {
        clusters.set(closestCentroidIdx, [])
      }
      clusters.get(closestCentroidIdx)?.push(i)
    }
    return clusters
  }

  // Relocate new centroids
  private updateCentroids(clusters: Map<number, number[]>) {
    let updated = false
    // Recalculate centroids
    for (let i = 0; i < this.centroids.length; i++) {
      const clusterData = clusters.get(i)?.map((index) => this.dataset[index])
      if (clusterData && clusterData.length > 0) {
        const mean = clusterData
          .reduce((acc, curr) => {
            return acc.map((d, i) => d + curr[i])
          })
          .map((d) => d / clusterData.length)
        if (eucledian(mean, this.centroids[i]) !== 0) {
          this.centroids[i] = mean
          updated = true
        }
      }
    }

    return updated
  }

  /**
   * Fits dataset into clusters.
   *
   * @param config KMeans Algorithm config
   */
  fit() {
    // Select random centroids
    this.centroids = !this.centroids.length
      ? this.selectRandomCentroids()
      : this.centroids

    // Perform KMeans clustering
    let clusters = new Map<number, number[]>()
    let updated = false
    this._epochs = 0

    do {
      clusters = this.assignToNearbyCentroids()
      updated = this.updateCentroids(clusters)
      this._epochs++
    } while (updated)

    return Array.from(clusters.values())
  }
}
