package pl.edu.agh.locust.algorithm

import breeze.linalg.DenseVector
import breeze.linalg.norm
import java.awt.Color

import pl.edu.agh.xinuk.algorithm.PlanCreator
import pl.edu.agh.xinuk.algorithm.{Metrics, Plans, Plan}
import pl.edu.agh.xinuk.model.{CellContents, CellId, CellState, Direction}
import pl.edu.agh.xinuk.model.grid.GridCellId
import pl.edu.agh.xinuk.model.grid.GridDirection
import pl.edu.agh.xinuk.model.grid.GridDirection._

import pl.edu.agh.locust.algorithm.ParticleAgentUpdate._
import pl.edu.agh.locust.model.{AgentContainer, AgentBehaviour, ParticleAgent}
import pl.edu.agh.locust.config.ParticleAgentConfig
import quadtree.{QuadTree, Point}

final case class ParticleAgentPlanCreator() extends PlanCreator[ParticleAgentConfig] {
  def createPlans(
      iteration: Long,
      cellId: CellId,
      cellState: CellState,
      neighbourContents: Map[Direction, CellContents]
  )(implicit config: ParticleAgentConfig): (Plans, Metrics) = {
    createContainerPlans(
      iteration,
      cellId.asInstanceOf[GridCellId],
      cellState.contents.asInstanceOf[AgentContainer[ParticleAgent]],
      neighbourContents.asInstanceOf[Map[GridDirection, AgentContainer[ParticleAgent]]]
    )
  }

  private def createContainerPlans[A <: ParticleAgent](
      iteration: Long,
      cellId: GridCellId,
      agentContainer: AgentContainer[A],
      neighbourContents: Map[GridDirection, AgentContainer[A]]
  )(implicit config: ParticleAgentConfig): (Plans, Metrics) = {
    val temporaryQuadtree = new QuadTree[A](
      center = Point(
        agentContainer.xMin + 0.5 * agentContainer.size,
        agentContainer.yMin + 0.5 * agentContainer.size
      ),
      halfDim = 1.5 * agentContainer.size + 1e-7
    )

    val neighbourAgents = neighbourContents
      .map({ case (direction, contents) =>
        createWrappedAgentViews(cellId, direction, contents).foreach(agent =>
          temporaryQuadtree.insert(Point(agent.position(0), agent.position(1)), agent)
        )
      })

    val localAgents = agentContainer.agents
    // val agentsInRange = neighbourAgents ++ localAgents

    localAgents.foreach(agent =>
      temporaryQuadtree.insert(Point(agent.position(0), agent.position(1)), agent)
    )

    val worldWidth = config.worldWidth * config.agentContainerSize
    val worldHeight = config.worldHeight * config.agentContainerSize

    val movedLocalAgents =
      localAgents
        .map(agent => {
          val conspecifics = temporaryQuadtree
            .knnSearch(Point(agent.position(0), agent.position(1)), config.occlusionThreshold + 1)
            .map(_._2)
            .filter(_ != agent)
          val updatedAgent = agentContainer.behaviour.update(agent, conspecifics)
          val movedAgent = agentContainer.behaviour.move(updatedAgent, config.timestepDuration)

          movedAgent
        })
        .groupMap(agent => {
          val xShift =
            if (agent.position(0) < agentContainer.xMin) -1
            else if (agent.position(0) > agentContainer.xMin + agentContainer.size) 1
            else 0

          val yShift =
            if (agent.position(1) < agentContainer.yMin) -1
            else if (agent.position(1) > agentContainer.yMin + agentContainer.size) 1
            else 0

          (xShift, yShift) match {
            case (0, 0)   => None
            case (-1, 0)  => Some(Top)
            case (-1, 1)  => Some(TopRight)
            case (0, 1)   => Some(Right)
            case (1, 1)   => Some(BottomRight)
            case (1, 0)   => Some(Bottom)
            case (1, -1)  => Some(BottomLeft)
            case (0, -1)  => Some(Left)
            case (-1, -1) => Some(TopLeft)
          }

        })(agent => {
          val agentX = ((agent.position(0) % worldWidth) + worldWidth) % worldWidth
          val agentY = ((agent.position(1) % worldHeight) + worldHeight) % worldHeight

          agentContainer.behaviour.translate(agent, DenseVector(agentX, agentY))

        })
        .updatedWith(None)(_.orElse(Some(Set.empty[A])))
        .map({ case (direction, agents) => (direction, Plan(AddAgents(agents))) })
        .toSeq

    // val plans = Plans(None -> Plan(AddAgents(movedLocalAgents)))
    // val plans = Plans(redistributeAgents(agentContainer, movedLocalAgents))
    val plans = Plans(movedLocalAgents)

    (plans, computeMetrics(localAgents, temporaryQuadtree))
    // (Plans.empty, ParticleAgentMetrics.empty)

  }

  private def createWrappedAgentViews[A <: ParticleAgent](
      currentCellId: GridCellId,
      neighbourDirection: GridDirection,
      agentContainer: AgentContainer[A]
  )(implicit config: ParticleAgentConfig): Iterable[A] = {
    val horizontalWrapAroundDistance = config.worldWidth * config.agentContainerSize

    val horizontalTranslation =
      if (neighbourDirection.of(currentCellId).x == -1) {
        -horizontalWrapAroundDistance
      } else if (neighbourDirection.of(currentCellId).x == config.worldWidth) {
        horizontalWrapAroundDistance
      } else 0.0

    val verticalWrapAroundDistance = config.worldHeight * config.agentContainerSize

    val verticalTranslation =
      if (neighbourDirection.of(currentCellId).y == -1) {
        -verticalWrapAroundDistance
      } else if (neighbourDirection.of(currentCellId).y == config.worldHeight) {
        verticalWrapAroundDistance
      } else 0.0

    if (horizontalTranslation != 0.0 || verticalTranslation != 0.0) {
      val translationVector = DenseVector(horizontalTranslation, verticalTranslation)

      agentContainer.agents.map(agent => {
        agentContainer.behaviour.translate(agent, agent.position + translationVector)
      })
    } else {
      agentContainer.agents
    }

  }

  // private def getAgentsInRange()

  private def computeMetrics[A <: ParticleAgent](
      agents: Iterable[A],
      temporaryQuadtree: QuadTree[A]
  )(implicit config: ParticleAgentConfig): ParticleAgentMetrics = {
    if (agents.size == 0) return ParticleAgentMetrics.empty

    val neighbourhoodOrders = agents
      .map(agent => {
        norm(
          temporaryQuadtree
            .knnSearch(Point(agent.position(0), agent.position(1)), config.localOrderNeighbours + 1)
            .map(_._2.direction)
            .reduce(_ + _)
        ) / (config.localOrderNeighbours + 1)
      })

    val localOrder = neighbourhoodOrders.sum / neighbourhoodOrders.size

    val directionSum = agents.map(_.direction).reduce(_ + _)

    val cellOrder = norm(directionSum) / agents.size

    ParticleAgentMetrics.init(
      agents.size,
      agents.size.toDouble / config.agentAmount,
      localOrder,
      cellOrder,
      directionSum
    )
  }

  private def redistributeAgents[A <: ParticleAgent](
      agentContainer: AgentContainer[A],
      agents: Set[A]
  ): Seq[(Option[Direction], Plan)] = {
    agents
      .groupBy(agent => {
        val xShift =
          if (agent.position(0) < agentContainer.xMin) -1
          else if (agent.position(0) > agentContainer.xMin + agentContainer.size) 1
          else 0

        val yShift =
          if (agent.position(1) < agentContainer.yMin) -1
          else if (agent.position(1) > agentContainer.yMin + agentContainer.size) 1
          else 0

        (xShift, yShift) match {
          case (0, 0)   => None
          case (-1, 0)  => Some(Top)
          case (-1, 1)  => Some(TopRight)
          case (0, 1)   => Some(Right)
          case (1, 1)   => Some(BottomRight)
          case (1, 0)   => Some(Bottom)
          case (1, -1)  => Some(BottomLeft)
          case (0, -1)  => Some(Left)
          case (-1, -1) => Some(TopLeft)
        }

      })
      .updatedWith(None)(_.orElse(Some(Set.empty[A])))
      .map({ case (direction, agents) => (direction, Plan(AddAgents(agents))) })
      .toSeq
  }
}
