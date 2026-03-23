lazy val xinukCore = ProjectRef(file("deps/xinuk"), "xinuk-core")

lazy val locustSimulation = (project in file("."))
  .settings(
    name := "locust-simulation",
    organization := "pl.edu.agh",
    version := "0.1.0",
    scalaVersion := "2.13.17",
    mainClass := Some("locust.LocustMain")
  )
  .dependsOn(xinukCore)
