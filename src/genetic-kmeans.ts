import { eucledian, KMeans } from './kmeans'

/**
 * GeneticKMeansConfig data structure.
 *
 * Store configurations for the GeneticKMeans
 */
export interface GeneticKMeansConfig {
  populationSize: number
  numGenerations: number
  kMin: number
  kMax: number
}

/**
 * Find the best solution for the KMeans algorithm
 */
export class GeneticKMeans {
  private readonly populationSize: number
  private readonly numGenerations: number
  private readonly data: number[][]
  private readonly kMin: number
  private readonly kMax: number

  constructor(data: number[][], config: GeneticKMeansConfig) {
    this.populationSize = config.populationSize
    this.numGenerations = config.numGenerations
    this.data = data
    this.kMin = config.kMin
    this.kMax = config.kMax
  }

  private initializePopulation(): number[][][] {
    const population: number[][][] = []
    for (let i = 0; i < this.populationSize; i++) {
      const k =
        Math.floor(Math.random() * (this.kMax - this.kMin + 1)) + this.kMin
      const centroids: number[][] = []
      for (let j = 0; j < k; j++) {
        const centroidIndex = Math.floor(Math.random() * this.data.length)
        centroids.push(this.data[centroidIndex])
      }
      population.push(centroids)
    }
    return population
  }

  private evaluateFitness(centroids: number[][]): number {
    const kmeans = new KMeans(centroids.length, this.data, centroids)
    const clusterIndecies = kmeans.fit()
    const clusters = clusterIndecies.map((indices) =>
      indices.map((index) => this.data[index])
    )
    let sumSquaredDistances = 0
    for (let i = 0; i < clusters.length; i++) {
      const centroid = centroids[i]
      for (let j = 0; j < clusters[i].length; j++) {
        const dataPoint = clusters[i][j]

        if (dataPoint == null || centroid == null) continue

        sumSquaredDistances += eucledian(dataPoint, centroid) ** 2
      }
    }
    return 1 / sumSquaredDistances
  }

  private selectParents(population: number[][][]): number[][][] {
    const parents: number[][][] = []
    for (let i = 0; i < this.populationSize / 2; i++) {
      const index1 = Math.floor(Math.random() * this.populationSize)
      const index2 = Math.floor(Math.random() * this.populationSize)
      const parent1 = population[index1]
      const parent2 = population[index2]
      if (this.evaluateFitness(parent1) > this.evaluateFitness(parent2)) {
        parents.push(parent1)
      } else {
        parents.push(parent2)
      }
    }
    return parents
  }

  private crossover(parents: number[][][]): number[][][] {
    const offspring: number[][][] = []
    for (let i = 0; i < parents.length; i += 2) {
      const parent1 = parents[i]
      const parent2 = parents[i + 1]

      if (parent2 == null) continue

      const k = parent1.length
      const child: number[][] = []
      for (let j = 0; j < k; j++) {
        if (Math.random() < 0.5) {
          child.push(parent1[j])
        } else {
          child.push(parent2[j])
        }
      }
      offspring.push(child)
    }
    return offspring
  }

  private mutate(offspring: number[][][]): number[][][] {
    const mutatedOffspring: number[][][] = []
    for (let i = 0; i < offspring.length; i++) {
      const child = offspring[i]
      const k = child.length
      for (let j = 0; j < k; j++) {
        if (Math.random() < 0.1) {
          const centroidIndex = Math.floor(Math.random() * this.data.length)
          child[j] = this.data[centroidIndex]
        }
        mutatedOffspring.push(child)
      }
    }
    return mutatedOffspring
  }

  public centroids(): number[][] {
    let population = this.initializePopulation()
    let bestCentroids: number[][] = []
    let bestFitness = 0
    for (let i = 0; i < this.numGenerations; i++) {
      const parents = this.selectParents(population)
      const offspring = this.crossover(parents)
      const mutatedOffspring = this.mutate(offspring)
      population = population.concat(mutatedOffspring)
      population = population
        .sort((a, b) => this.evaluateFitness(b) - this.evaluateFitness(a))
        .slice(0, this.populationSize)
      const bestIterationCentroids = population[0]
      const bestIterationFitness = this.evaluateFitness(bestIterationCentroids)
      if (bestIterationFitness > bestFitness) {
        bestCentroids = bestIterationCentroids
        bestFitness = bestIterationFitness
      }
    }
    return bestCentroids.filter((solution) => solution != null)
  }
}
