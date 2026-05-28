package pl.edu.agh.locust.model

import java.awt.Color
import breeze.linalg.{DenseVector, norm, normalize, sum}
import pl.edu.agh.xinuk.model.CellContents
import pl.edu.agh.xinuk.config.XinukConfig
import pl.edu.agh.locust.config.SPPAgentConfig
import scala.collection.mutable.PriorityQueue

sealed trait Agent {
  val position: DenseVector[Double]
  val direction: DenseVector[Double]
}

sealed trait AgentBehaviour[A <: Agent, C <: XinukConfig] {
  def update(agent: A, conspecifics: Iterable[A])(implicit config: C): A
  def move(agent: A, deltaTime: Double)(implicit config: C): A
  def translate(agent: A, newPosition: DenseVector[Double]): A
}

final case class SPPAgent(position: DenseVector[Double], direction: DenseVector[Double])
    extends Agent

object SPPAgent {
  implicit case object Behaviour extends AgentBehaviour[SPPAgent, SPPAgentConfig] {
    def update(agent: SPPAgent, others: Iterable[SPPAgent])(implicit
        config: SPPAgentConfig
    ): SPPAgent = {
      if (others.size == 0) return agent
      val socialForce: DenseVector[Double] = others
        .map(other => {

          val displacement = agent.position - other.position
          val distance = norm(displacement)
          val displacementDirection = normalize(displacement)

          if (distance < config.repulsionRange)
            -config.repulsionWeight * displacementDirection
          else if (distance < config.alignmentRange)
            config.alignmentWeight * other.direction
          else if (distance < config.attractionRange)
            config.attractionWeight * displacementDirection
          else
            DenseVector.zeros[Double](2)

        })
        .reduce(_ + _)

      val newDirection: DenseVector[Double] =
        normalize(
          config.previousDirectionWeight * agent.direction +
            (1 - config.previousDirectionWeight) * normalize(socialForce)
        )

      agent.copy(direction = newDirection)
    }

    def move(agent: SPPAgent, deltaTime: Double)(implicit config: SPPAgentConfig): SPPAgent = {
      val newPosition = agent.position + agent.direction * deltaTime * config.averageSpeed

      agent.copy(position = newPosition)
    }

    def translate(agent: SPPAgent, newPosition: DenseVector[Double]): SPPAgent = {
      agent.copy(position = newPosition)
    }

    implicit class IterableOps[T](val elements: Iterable[T]) extends AnyVal {
      def takeBy[A](n: Int, byFunc: T => A)(implicit
          ord: Ordering[A]
      ): List[T] = {

        // 1. Define ordering to ONLY look at the first element of the tuple
        implicit val ByOrdering: Ordering[(A, T)] = Ordering.by(_._1)

        // 2. PriorityQueue is a Max-Heap by default in Scala.
        // The largest '_._1' will be at the head of the queue.
        val maxHeap = PriorityQueue.empty[(A, T)]

        // 3. Iterate the original set, applying the map function on the fly
        elements.foreach { item =>
          val mappedTuple = (byFunc(item), item) // Transform right before evaluating

          maxHeap.enqueue(mappedTuple)

          // 4. If we exceed n, discard the element with the LARGEST first value
          if (maxHeap.size > n) {
            maxHeap.dequeue()
          }
        }

        // 5. Extract results and sort them ascending (lowest to highest)
        maxHeap.map(_._2).toList
      }
    }
  }
}

final case class AgentContainer[A <: Agent](
    var agents: Iterable[A],
    var lastUpdateIteration: Long,
    behaviour: AgentBehaviour[A, SPPAgentConfig],
    size: Double,
    xMin: Double,
    yMin: Double,
    particlesColor: Color
) extends CellContents
