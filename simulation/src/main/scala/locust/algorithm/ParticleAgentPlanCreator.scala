package pl.edu.agh.locust.algorithm

import breeze.linalg.DenseVector
import java.awt.Color

import pl.edu.agh.xinuk.algorithm.PlanCreator
import pl.edu.agh.xinuk.algorithm.{Metrics, Plans, Plan}
import pl.edu.agh.xinuk.model.{CellContents, CellId, CellState, Direction}
import pl.edu.agh.xinuk.model.grid.GridCellId
import pl.edu.agh.xinuk.model.grid.GridDirection
import pl.edu.agh.xinuk.model.grid.GridDirection._

import pl.edu.agh.locust.algorithm.ParticleAgentUpdate._
import pl.edu.agh.locust.model.{AgentContainer, AgentBehaviour, Agent}
import pl.edu.agh.locust.config.SPPAgentConfig
import pl.edu.agh.locust.model.SPPAgent
import quadtree.Point
import quadtree.QuadTree

final case class ParticleAgentPlanCreator() extends PlanCreator[SPPAgentConfig] {
  def createPlans(
      iteration: Long,
      cellId: CellId,
      cellState: CellState,
      neighbourContents: Map[Direction, CellContents]
  )(implicit config: SPPAgentConfig): (Plans, Metrics) = {
    createContainerPlans(
      iteration,
      cellId.asInstanceOf[GridCellId],
      cellState.contents.asInstanceOf[AgentContainer[_ <: Agent]],
      neighbourContents
    )
  }

  private def createContainerPlans[A <: Agent](
      iteration: Long,
      cellId: GridCellId,
      agentContainer: AgentContainer[A],
      neighbourContents: Map[Direction, CellContents]
  )(implicit config: SPPAgentConfig): (Plans, Metrics) = {

    val neighbourAgents = neighbourContents
      .map({ case (direction, contents) =>
        createWrappedAgentViews(
          cellId,
          direction.asInstanceOf[GridDirection],
          contents.asInstanceOf[AgentContainer[A]]
        )
      })

    val localAgents = agentContainer.agents
    // val agentsInRange = neighbourAgents ++ localAgents

    val worldWidth = config.worldWidth * config.agentContainerSize
    val worldHeight = config.worldHeight * config.agentContainerSize

    // val movedLocalAgents: Set[A] =
    val movedLocalAgents =
      agentContainer.agents
        .map(agent => {
          // println(agent.position)
          // println(cellId)
          val localNearestAgents = agentContainer.knnSearch(agent, config.occlusionThreshold)
          val updatedAgent = agentContainer.behaviour.update(agent, localNearestAgents)
          val movedAgent = agentContainer.behaviour.move(updatedAgent, config.timestepLength)

          movedAgent

          // val agentX = ((movedAgent.position(0) % worldWidth) + worldWidth) % worldWidth
          // val agentY = ((movedAgent.position(1) % worldHeight) + worldHeight) % worldHeight
          // println(agentX, agentY)
          //
          // agentContainer.behaviour.translate(movedAgent, DenseVector(agentX, agentY))

        })
        .toSet[A]
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

    (plans, ParticleAgentMetrics.empty)
    // (Plans.empty, ParticleAgentMetrics.empty)

  }

  private def createWrappedAgentViews[A <: Agent](
      currentCellId: GridCellId,
      neighbourDirection: GridDirection,
      agentContainer: AgentContainer[A]
  )(implicit config: SPPAgentConfig): AgentContainer[A] = {
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

    if (horizontalTranslation != 0.0 && verticalTranslation != 0.0) {
      val translationVector = DenseVector(horizontalTranslation, verticalTranslation)
      val view = agentContainer.copy(
        xMin = agentContainer.xMin + horizontalTranslation,
        yMin = agentContainer.yMin + verticalTranslation
      )
      view.clear()

      agentContainer.agents
        .foreach(agent => {
          val translatedAgent =
            agentContainer.behaviour.translate(agent, agent.position + translationVector)
          view.insert(translatedAgent)
          // q.insert(Point(translatedAgent.position(0), translatedAgent.position(1)), translatedAgent)
        })

      view

    } else agentContainer

  }

  private def redistributeAgents[A <: Agent](
      agentContainer: AgentContainer[A],
      agents: Iterable[A]
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
