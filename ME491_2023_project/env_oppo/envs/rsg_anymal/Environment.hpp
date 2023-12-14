// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
// raisim include
#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"


#include "../../Yaml.hpp"
#include "../../BasicEigenTypes.hpp"
#include "../../Reward.hpp"

#include "AnymalController_oppo_1.hpp"

namespace raisim{

class ENVIRONMENT {

 public:

  explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable) :
      visualizable_(visualizable) {
    /// add objects
    robot_ = world_.addArticulatedSystem(resourceDir + "/anymal/urdf/anymal_red.urdf");
    robot_->setName(PLAYER_NAME);
    controller_.setName(PLAYER_NAME);
    robot_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

    auto* ground = world_.addGround();
    ground->setName("ground");

    controller_.setName(PLAYER_NAME);
    controller_.setOpponentName(OPPONENT_NAME);
    controller_.create(&world_);
    READ_YAML(double, simulation_dt_, cfg["simulation_dt"])
    READ_YAML(double, control_dt_, cfg["control_dt"])

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

    /// visualize if it is the first environment
    if (visualizable_) {
      const int port = 8085;
      server_ = std::make_unique<raisim::RaisimServer>(&world_);
      server_->launchServer(port);
      server_->focusOn(robot_);
      auto cage = server_->addVisualCylinder("cage", 3.0, 0.05);
      cage->setPosition(0,0,0);
      targetSphere = server_->addVisualSphere("targetSphere", 0.05, 1, 0, 0, 1);
    }
    factor_ = 0;
    strategy = 0;
    maxForce = 180;
    maxTorque = 3;
    maxJointTorque = 0.35;
    randomFactor_.setZero();
    gc_.setZero(robot_->getGeneralizedCoordinateDim());
    gv_.setZero(robot_->getDOF() + 6);
    extGenF.setZero(robot_->getDOF() + 6);
  }

  void init() {}

  double square(double a){
    return a * a;
  }

  Eigen::Vector3d quatToRPY(const Eigen::Vector4d& q) {
    Eigen::Vector3d rpy;
    const double as = std::min(-2. * (q[1] * q[3] - q[0] * q[2]), .99999);
    rpy(2) = std::atan2(2 * (q[1] * q[2] + q[0] * q[3]),
                        square(q[0]) + square(q[1]) - square(q[2]) - square(q[3]));
    rpy(1) = std::asin(as);
    rpy(0) = std::atan2(2 * (q[2] * q[3] + q[0] * q[1]),
                        square(q[0]) - square(q[1]) - square(q[2]) + square(q[3]));
    return rpy; 
  }

  Eigen::Vector4d randomQuatGen(){
    const double u = 0.5 + 0.5*uniDist_(gen_);
    const double v = 0.5 + 0.5*uniDist_(gen_);
    const double w = 0.5 + 0.5*uniDist_(gen_);

    Eigen::Vector4d q;
    q(0) = sqrt(1 - u) * sin(2*M_PI*v);
    q(1) = sqrt(1 - u) * cos(2*M_PI*v);
    q(2) = sqrt(u) * sin(2*M_PI*w);
    q(3) = sqrt(u) * cos(2*M_PI*w);
    return q;
  }

  void reset() {
    timer_ = 0;
    int playerNum = 1*(uniDist_(gen_) > 0.0);
    controller_.setPlayerNum(playerNum);
    double theta = uniDist_(gen_) * 2 * M_PI;
    controller_.reset(&world_, theta);

    double p = abs(uniDist_(gen_));
    if(p < 0.2 || factor_ == 0){
      strategy = 0;
      extF.setZero();
      extT.setZero();
      extGenF.setZero();
    }
    else{
      extF << uniDist_(gen_), uniDist_(gen_), uniDist_(gen_);
      extF.cwiseProduct(Eigen::Vector3d(1, 1, 0.3).normalized());
      extF *= maxForce * factor_;
      extT << uniDist_(gen_), uniDist_(gen_), uniDist_(gen_);
      extT.cwiseProduct(Eigen::Vector3d(1, 1, 2).normalized());
      extT *= maxTorque * factor_;
      if(p < 0.4) strategy = 1;
      else if(p < 0.6) strategy = 2;
      else if(p < 0.8) strategy = 3;
      else{
        strategy = 4;
        Eigen::VectorXd extJointTorque(12);
        for(int i = 0; i < 12; i++){
          extJointTorque(i) = uniDist_(gen_);
        }
        extGenF << extF, extT, maxJointTorque*factor_*extJointTorque;
      }
    }
    // randomFactor_ << 0.75 + 0.25*uniDist_(gen_), 0.8*normDist_(gen_), uniDist_(gen_);

    // reset states
    robot_->getState(gc_, gv_);
  }

  void reset2() {
  }


  float step(const Eigen::Ref<EigenVec> &action) {
    timer_++;
    controller_.advance(&world_, action);

    double multiplier = 1 + 0.1*normDist_(gen_);
    if(strategy == 0){
      // no external force
    }
    else if(strategy == 1){ // external force
      robot_->setExternalForce(0, extF*multiplier);
    }
    else if(strategy == 2){ // external torque
      robot_->setExternalTorque(0, extT*multiplier);
    }
    else if(strategy == 3){ // external force and torque
      robot_->setExternalForce(0, extF*multiplier);
      robot_->setExternalTorque(0, extT*multiplier);
    }
    else if(strategy == 4){ // general external force and torque
      robot_->setGeneralizedForce(extGenF*multiplier);
    }
    else{
      std::cout << "strategy error" << std::endl;
    }


    for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
      if (server_) server_->lockVisualizationServerMutex();
      world_.integrate();
      if (server_) server_->unlockVisualizationServerMutex();
    }
    
    controller_.recordReward(&rewards_);
    if(visualizable_){
      Eigen::Vector2d pos = controller_.getTargetPos();
      targetSphere->setPosition(pos(0), pos(1), 0.1);
    }
    return exp(rewards_.sum());
  }

  float step2(const Eigen::Ref<EigenVec> &action, const Eigen::Ref<EigenVec> &actionO) {
    return 0.f;
  }

  void observe(Eigen::Ref<EigenVec> ob) {
    controller_.updateObservation(&world_);
    ob = controller_.getObservation().cast<float>();
  }

  void observe2(Eigen::Ref<EigenVec> obO) {
  }

  void printTerminalInfo(const char* terminalType){
    std::cout << terminalType << ", step = " << timer_ << std::endl;
  }

  bool isTerminalState(float &terminalReward) {
    robot_->getState(gc_, gv_);

    if(controller_.isTerminalState(&world_)){
      printTerminalInfo("controller termianl");
      // terminalReward = -20;
      terminalReward = -700;
      return true;
    }
    if(gc_.head(2).norm() > 3){
      printTerminalInfo("out of bound");
      // terminalReward = -20;
      terminalReward = -700; //-200 -> -500 -> -700
      return true;
    }
    if (timer_ > 20 * 100) {
      terminalReward = 0;
      return true;
    }
    return false;
  }

  void curriculumUpdate(double factor){
    factor_ = std::min(factor, 1.0);
  }

  void close() { if (server_) server_->killServer(); };

  void setSeed(int seed) {};

  void setSimulationTimeStep(double dt) {
    simulation_dt_ = dt;
    world_.setTimeStep(dt);
  }
  void setControlTimeStep(double dt) { control_dt_ = dt; }

  int getObDim() { return controller_.getObDim(); }

  int getActionDim() { return controller_.getActionDim(); }

  double getControlTimeStep() { return control_dt_; }

  double getSimulationTimeStep() { return simulation_dt_; }

  raisim::World *getWorld() { return &world_; }

  void turnOffVisualization() { server_->hibernate(); }

  void turnOnVisualization() { server_->wakeup(); }

  void startRecordingVideo(const std::string &videoName) { server_->startRecordingVideo(videoName); }

  void stopRecordingVideo() { server_->stopRecordingVideo(); }

  raisim::Reward& getRewards() { return rewards_; }

 private:
  int timer_ = 0;
  int player1_win = 0, player2_win = 0, draw = 0;
  bool visualizable_ = false;
  double terminalRewardCoeff_ = -10.;
  raisim::ArticulatedSystem *robot_;
  // AnymalController_20002000 controller_;
  AnymalController_20002001 controller_;
  raisim::Visuals* targetSphere;
  raisim::World world_;
  raisim::Reward rewards_;
  double simulation_dt_ = 0.001;
  double control_dt_ = 0.01;
  Eigen::VectorXd gc_, gv_;
  Eigen::VectorXd gcOppo_, gvOppo_;

  // opponent object
  int strategy;
  Eigen::Vector3d randomFactor_;
  // external force
  double factor_;
  double maxForce, maxTorque, maxJointTorque;
  Eigen::Vector3d extF, extT;
  Eigen::VectorXd extGenF;

  std::unique_ptr<raisim::RaisimServer> server_;
  thread_local static std::normal_distribution<double> normDist_;
  thread_local static std::uniform_real_distribution<double> uniDist_;
  thread_local static std::mt19937 gen_;
};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
thread_local std::normal_distribution<double> raisim::ENVIRONMENT::normDist_(0., 1.);
thread_local std::uniform_real_distribution<double> raisim::ENVIRONMENT::uniDist_(0., 1.);
}
