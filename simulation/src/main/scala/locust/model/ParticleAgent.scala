package pl.edu.agh.locust.model

import java.awt.Color
import breeze.linalg.{DenseVector, norm, normalize, sum}
import pl.edu.agh.xinuk.model.CellContents
import pl.edu.agh.xinuk.config.XinukConfig
import pl.edu.agh.locust.config.SPPAgentConfig
import scala.collection.mutable.PriorityQueue
import quadtree.{QuadTree, Point}

trait Agent {
  val position: DenseVector[Double]
}

trait AgentBehaviour[A <: Agent, C <: XinukConfig] {
  def update(agent: A, conspecifics: Iterable[A])(implicit config: C): A
  def move(agent: A, deltaTime: Double)(implicit config: C): A
  def translate(agent: A, newPosition: DenseVector[Double]): A
}

final case class AgentContainer[A <: Agent](
    var lastUpdateIteration: Long,
    behaviour: AgentBehaviour[A, SPPAgentConfig],
    size: Double,
    xMin: Double,
    yMin: Double,
    particlesColor: Color
) extends CellContents {
  private var _agents: QuadTree[A] = createQuadTree()

  private def createQuadTree(): QuadTree[A] = {
    val containerXCenter = xMin + 0.5 * size
    val containerYCenter = yMin + 0.5 * size

    new QuadTree[A](
      2,
      Point(containerXCenter, containerYCenter),
      size / 2 + 1e-9
    )
  }

  def clear(): Unit = {
    _agents = createQuadTree()
  }

  def insert(agent: A): Boolean = {
    _agents.insert(Point(agent.position(0), agent.position(1)), agent)
  }

  def agents: Iterable[A] = {
    _agents
      .rangeSearch(
        Point(xMin - 1e-7, yMin + 1e-7),
        Point(xMin + size + 1e-7, yMin + size + 1e-7)
      )
      .map(_._2)
  }

  def knnSearch(agent: A, k: Integer): Iterable[A] = {
    _agents.knnSearch(Point(agent.position(0), agent.position(1)), k).map(_._2)
  }
}
