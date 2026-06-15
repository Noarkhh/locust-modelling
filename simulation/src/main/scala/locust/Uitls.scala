package pl.edu.agh.locust.utils
import breeze.linalg.{DenseVector, norm, normalize, sum}

object ImplicitVectorOps {
  implicit class DenseVectorOps(v: DenseVector[Double]) {
    def normalize(): DenseVector[Double] = breeze.linalg.normalize(v)
    def norm(): Double = breeze.linalg.norm(v)
  }
}
