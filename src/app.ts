import * as path from 'path'
import { feathers } from '@feathersjs/feathers'
import { GeneticKMeans, GeneticKMeansConfig } from './genetic-kmeans'
import {
  bodyParser,
  errorHandler,
  koa,
  rest,
  serveStatic
} from '@feathersjs/koa'
import { KMeans } from './kmeans'

// Configs
const STATIC_RESOURCES = path.join(__dirname, '..', 'public')

// Payload for the GeneticKMeansService.fit
interface ClusterPayload<T> {
  meta: GeneticKMeansConfig
  payload: T[]
  features: string[]
}

// Create a service that we will expose over REST Api
class ClusterService {
  async create(data: ClusterPayload<any>) {
    // Extract features from the payload
    const dataset = data.payload.map((entry) => {
      return Object.entries(entry)
        .filter(([key, _]) => data.features.includes(key))
        .map(([_, value]) => parseFloat(value as string))
    })
    // Just return all our messages
    const gkmeans = new GeneticKMeans(dataset, data.meta)
    const centroids = gkmeans.centroids()
    const kmeans = new KMeans(centroids.length, dataset, centroids)
    const clusteredIndecies = kmeans.fit()
    // Get clusters
    const clusters = clusteredIndecies.map((indecies) =>
      indecies.map((index) => data.payload[index])
    )
    return { centroids, clusters }
  }
}

// This tells TypeScript what services we are registering
type ServiceTypes = {
  clusterService: ClusterService
}

// Create a KoaJS compatible feather's app
const app = koa<ServiceTypes>(feathers())

// Use public folder for hosting static resources

// Register the GeneticKMeans service on the Feathers application
// @ts-ignore
app.use(errorHandler())

// @ts-ignore
app.use(serveStatic(STATIC_RESOURCES))

// @ts-ignore
app.use(bodyParser())

// @ts-ignore
app.configure(rest())

// Register cluster service
app.use('clusterService', new ClusterService())

// Log every time a new fitting request
app.service('clusterService').on('fit', (payload: ClusterPayload<any>) => {
  console.log(
    `Fitting using GeneticKMeans algorithm with ${payload.payload.length} samples...`
  )
})

app
  .listen(3030)
  .then(() => console.log('Clustering service is runnign on localhost:3030'))
