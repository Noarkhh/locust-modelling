lazy val xinukCore = ProjectRef(file("deps/xinuk"), "xinuk-core")

lazy val locustSimulation = (project in file("."))
  .settings(
    name := "locust-simulation",
    organization := "pl.edu.agh",
    version := "0.1.0",
    scalaVersion := "2.13.17",
    Compile / mainClass := Some("pl.edu.agh.locust.LocustMain"),
    run / fork := true,
    libraryDependencies ++= Seq(
      "org.scalanlp" %% "breeze" % "2.1.0",
      "ch.qos.logback" % "logback-classic" % "1.2.3"
    )
  )
  .dependsOn(xinukCore)
