package pl.edu.agh.locust.algorithm

import pl.edu.agh.xinuk.algorithm.Metrics
import breeze.linalg.DenseVector
import breeze.linalg.norm

final case class ParticleAgentMetrics(
    locustCount: Long,
    locustPositions: Seq[DenseVector[Double]],
    locustVelocities: Seq[DenseVector[Double]]
) extends Metrics {

  override def log: String = {
    val avgSpeed = locustVelocities.map(norm(_)).sum
    s"$locustCount;$avgSpeed"
  }

  override def series: Vector[(String, Double)] = Vector(
    "Locusts" -> locustCount.toDouble,
    "AvgSpeed" -> locustVelocities.map(norm(_)).sum
  )

  override def +(other: Metrics): ParticleAgentMetrics = {
    other match {
      case ParticleAgentMetrics(otherLocustCount, otherLocustPositions, otherLocustVelocities) =>
        ParticleAgentMetrics(
          locustCount + otherLocustCount,
          otherLocustPositions ++ locustPositions,
          otherLocustVelocities ++ locustVelocities
        )
    }
  }
}

object ParticleAgentMetrics {
  val MetricHeaders = Vector(
    "locustCount",
    "avgSpeed"
  )
}
