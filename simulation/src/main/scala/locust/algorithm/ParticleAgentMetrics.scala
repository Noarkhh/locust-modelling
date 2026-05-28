package pl.edu.agh.locust.algorithm

import pl.edu.agh.xinuk.algorithm.Metrics
import breeze.linalg.DenseVector
import breeze.linalg.norm

final case class ParticleAgentMetrics(
    agentsCount: Int,
    populationShare: Double,
    localOrders: Iterable[Double],
    cellOrders: Iterable[Double],
    directionSum: DenseVector[Double]
) extends Metrics {

  override def log: String = {
    val averageLocalOrder = localOrders.sum / localOrders.size
    val averageCellOrder = cellOrders.sum / cellOrders.size
    val shardOrder = norm(directionSum) / agentsCount
    s"$populationShare;$averageLocalOrder;$averageCellOrder;$shardOrder;${directionSum(0)};${directionSum(1)}"
  }

  override def series: Vector[(String, Double)] = Vector(
    "populationShare" -> populationShare,
    "averageLocalOrder" -> localOrders.sum / localOrders.size,
    "averageCellOrder" -> cellOrders.sum / cellOrders.size,
    "shardOrder" -> norm(directionSum) / agentsCount

    // "directionSumX" -> directionSum(0),
    // "directionSumY" -> directionSum(1)
  )

  override def +(other: Metrics): ParticleAgentMetrics = {
    other match {
      case ParticleAgentMetrics(
            otherAgentsCount,
            otherPopulationShare,
            otherLocalOrders,
            otherCellOrders,
            otherDirectionSum
          ) =>
        ParticleAgentMetrics(
          agentsCount + otherAgentsCount,
          populationShare + otherPopulationShare,
          localOrders ++ otherLocalOrders,
          cellOrders ++ otherCellOrders,
          directionSum + otherDirectionSum
        )
    }
  }
}

object ParticleAgentMetrics {
  private val Empty =
    ParticleAgentMetrics(0, 0.0, Seq.empty, Seq.empty, DenseVector[Double](0.0, 0.0))

  def empty: ParticleAgentMetrics = Empty

  def init(
      agentsCount: Int,
      populationShare: Double,
      localOrder: Double,
      cellOrder: Double,
      directionSum: DenseVector[Double]
  ): ParticleAgentMetrics =
    ParticleAgentMetrics(
      agentsCount,
      populationShare,
      Vector(localOrder),
      Vector(cellOrder),
      directionSum
    )

  val MetricHeaders = Vector(
    "populationShare",
    "averageLocalOrder",
    "averageCellOrder",
    "shardOrder",
    "directionSumX",
    "directionSumY"
  )
}
