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
#include "raisim/object/singleBodies/SingleBodyObject.hpp"
#include "raisim/object/singleBodies/Box.hpp"
#include "raisim/object/singleBodies/Capsule.hpp"
#include "raisim/object/singleBodies/Cylinder.hpp"
#include "raisim/object/singleBodies/Sphere.hpp"

#include "../../Yaml.hpp"
#include "../../BasicEigenTypes.hpp"
#include "../../Reward.hpp"

#include TRAINING_HEADER_FILE_TO_INCLUDE
#include OPPONENT_HEADER_FILE_TO_INCLUDE

namespace raisim{

class ENVIRONMENT {

 public:

  explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable) :
      visualizable_(visualizable) {
    /// add objects
    robot_ = world_.addArticulatedSystem(resourceDir + "/anymal/urdf/anymal_red.urdf");
    robot_->setName(PLAYER_NAME);
    controller_.setName(PLAYER_NAME);
    controller_.setOpponentName(OPPONENT_NAME);
    robot_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

    robotO_ = world_.addArticulatedSystem(resourceDir + "/anymal/urdf/anymal_blue.urdf");
    robotO_->setName(OPPONENT_NAME);
    controllerO_self.setName(OPPONENT_NAME);
    controllerO_oppo.setName(OPPONENT_NAME);
    controllerO_self.setOpponentName(PLAYER_NAME);
    controllerO_oppo.setOpponentName(PLAYER_NAME);
    robotO_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

    mode_ = false;
    factor_ = 0.0;

    auto* ground = world_.addGround();
    ground->setName("ground");

    controller_.create(&world_);
    controllerO_self.create(&world_);
    controllerO_oppo.create(&world_);
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
    
    // add opponent
    objects_.push_back(world_.addSphere(0.1, 1.0));
    objects_.push_back(world_.addBox(0.1, 0.1, 0.1, 1.0));
    objects_.push_back(world_.addCylinder(0.1, 0.1, 1.0));
    objects_.push_back(world_.addCapsule(0.1, 0.1, 1.0));
    objIdx = 0;
    strategy = 0;
    randomFactor_.setZero();
    gc_.setZero(robot_->getGeneralizedCoordinateDim());
    gv_.setZero(robot_->getDOF() + 6);
    gcOppo_.setZero(robotO_->getGeneralizedCoordinateDim());
    gvOppo_.setZero(robotO_->getDOF() + 6);
    targetPos_.setZero();
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

  void reset() { // only for training without other opponent
    timer_ = 0;
    mode_ = true;
    const int playerNum = 1*(uniDist_(gen_) > 0.0);
    const double theta = uniDist_(gen_) * 2 * M_PI;

    // Set object, size, oreientation, properties
    std::uniform_int_distribution<int> objDist_(-5,3);
    objIdx = objDist_(gen_);
    // if(objIdx >= 0) objIdx = objDist_(gen_);

    const double mass = 35.0 + 15.0*uniDist_(gen_);
    double mu = 0.7 + 0.4*uniDist_(gen_);
    double height;
    if(objIdx < 0){ // robot
      minObjSize = 0.3;
    }
    else{ // object
      // remove previous object
      if(objects_[objIdx] != nullptr) {
        world_.removeObject(objects_[objIdx]);
      }
      // introduce bias in CoM
      if(objIdx == 0){ // shpere
        minObjSize = 0.5 + 0.35*uniDist_(gen_);
        height = minObjSize;
        objects_[objIdx] = world_.addSphere(minObjSize, mass);
        objects_[objIdx]->setName("sphere");
        world_.setMaterialPairProp("ground", "sphere", mu, 0.0, 0.01);
      }
      else if(objIdx == 1){ // box
        Eigen::Vector3d xyz;
        xyz << 0.75 + 0.45*uniDist_(gen_), 0.75 + 0.45*uniDist_(gen_), 0.75 + 0.45*uniDist_(gen_); // 0.3~1.2
        minObjSize = xyz.head(2).cwiseAbs().minCoeff();
        minObjSize *= 0.5;
        objects_[objIdx] = world_.addBox(xyz(0), xyz(1), xyz(2), mass);
        objects_[objIdx]->setName("box");
        world_.setMaterialPairProp("ground", "box", 0.02*mu, 0.0, 0.01);
      }
      else if(objIdx == 2){ // cylinder
        minObjSize = 0.5 + 0.35*uniDist_(gen_);
        height = 0.4 + 0.25*uniDist_(gen_);
        objects_[objIdx] = world_.addCylinder(minObjSize, 2*height, mass);
        objects_[objIdx]->setName("cylinder");
        world_.setMaterialPairProp("ground", "cylinder", 0.02*mu, 0.0, 0.01);
      }
      else if(objIdx == 3){ // capsule
        minObjSize = 0.5 + 0.35*uniDist_(gen_);
        height = minObjSize + 0.35 + 0.25*uniDist_(gen_);
        objects_[objIdx] = world_.addCapsule(minObjSize, 2*(height - minObjSize), mass);
        objects_[objIdx]->setName("capsule");
        world_.setMaterialPairProp("ground", "capsule", mu, 0.0, 0.01);
      }
    }

    // set obj position
    Eigen::Vector3d objPos = Eigen::Vector3d(gc_(0), gc_(1), 1.2); 
    const Eigen::Vector4d objOri = randomQuatGen();
    while((objPos.head(2) - gc_.head(2)).norm() < 1.5){
      const double r = 1.35*(1 + uniDist_(gen_));
      const double theta = 2*M_PI*uniDist_(gen_);
      objPos << r*cos(theta), r*sin(theta), 1.2;
    }
    if(objIdx < 0){ 
      objPos(2) = 0.6 + 0.15*uniDist_(gen_);
      controllerO_self.setPlayerNum(1 - playerNum);
      controllerO_self.resetGivenPos(&world_, objPos, gc_.head(3));
    }
    else{
      // set object pos
      objects_[objIdx]->setPosition(objPos(0), objPos(1), objPos(2));
      objects_[objIdx]->setOrientation(objOri(0), objOri(1), objOri(2), objOri(3));
      controllerO_self.resetFar(&world_);
    }
    for(int i = 0; i < 4; i++) if(i != objIdx) objects_[i]->setPosition(30, 30, -100);
    controller_.reset(&world_, theta, objIdx, objects_[objIdx]);
    //
    strategy = int(uniDist_(gen_) > 0); // 0 is fight, 1 is go to the center
    randomFactor_ << 0.75 + 0.25*uniDist_(gen_), 0.8*normDist_(gen_), uniDist_(gen_); // force mag, direction, disturbance

    // reset states
    robot_->getState(gc_, gv_);
    gcOppo_.head(3) = objPos;
    gvOppo_.setZero();
  }

  void reset2(bool mode) {
    mode_ = mode;
    timer_ = 0;
    objIdx = -1;
    if(uniDist_(gen_) > 0.8) objIdx = -100; //pdStanding mode with 10% prob
    minObjSize = 0.3;
    const int playerNum = 1*(uniDist_(gen_) > 0.0);
    const int playerNumOppo = 1 - playerNum;
    const double theta = uniDist_(gen_) * 2 * M_PI;
    controller_.setPlayerNum(playerNum);
    controller_.resetRandom(&world_);
    if(mode_){
      controllerO_self.setPlayerNum(playerNumOppo);
      controllerO_self.reset(&world_, theta, objIdx, NULL);
    }
    else{
      controllerO_oppo.setPlayerNum(playerNumOppo);
      controllerO_oppo.reset(&world_, theta, minObjSize, objIdx, NULL);
    }
    // reset states
    robot_->getState(gc_, gv_);
    robotO_->getState(gcOppo_, gvOppo_);
    for(int i = 0; i < 4; i++) objects_[i]->setPosition(30, 30, -100);
  }

  float step(const Eigen::Ref<EigenVec> &action) {
    // for(int i = 0; i < 12; i++){
    //   if(isnan(action(i))){
    //     std::cout << "action(" << i << ") is nan" << std::endl;
    //     return false;
    //   }
    // }
    if(objIdx < 0){
      Eigen::VectorXf oppoAction(12); oppoAction.setZero();
      oppoAction.setZero();
      return step2(action, oppoAction);
    }
    timer_++;
    controller_.advance(&world_, action);

    // external force on the object
    robot_->getState(gc_, gv_);
    controller_.getOppoState(gcOppo_, gvOppo_);
    raisim::Vec<3> externalForce;
    double forceMag = randomFactor_(0)*std::max(300 + 200*normDist_(gen_), 0.0);
    Eigen::Vector3d direction;
    if(strategy){ // fight
      direction = gc_.head(3) - gcOppo_.head(3);
      direction.normalized();
    }
    else{ // go to the center
      direction = -gcOppo_.head(3);
    }
    direction(2) = 0;
    direction.normalized();
    Eigen::Matrix3d directionPerturb;
    double theta = randomFactor_(1) + 0.8*normDist_(gen_);
    directionPerturb << cos(theta), -sin(theta), 0,
                        sin(theta),  cos(theta), 0,
                                0,           0, 1;
    direction = directionPerturb*direction;
    externalForce[0] = forceMag*direction(0);
    externalForce[1] = forceMag*direction(1);
    externalForce[2] = 0;
    objects_[objIdx]->setExternalForce(0, externalForce);
    // robot_->setExternalForce(0, -externalForce);
    for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
      if (server_) server_->lockVisualizationServerMutex();
      world_.integrate();
      if (server_) server_->unlockVisualizationServerMutex();
    }
    
    controller_.recordReward(&rewards_);
    return exp(rewards_.sum());
  }

  float step2(const Eigen::Ref<EigenVec> &action, const Eigen::Ref<EigenVec> &actionO) {
    timer_++;
    controller_.advance(&world_, action);
    if(mode_) controllerO_self.advance(&world_, actionO);
    else      controllerO_oppo.advance(&world_, actionO);
    //
    for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
      if (server_) server_->lockVisualizationServerMutex();
      world_.integrate();
      if (server_) server_->unlockVisualizationServerMutex();
    }
    
    controller_.updateObservation(&world_);
    if(mode_) controllerO_self.updateObservation(&world_);
    else      controllerO_oppo.updateObservation(&world_);
    controller_.recordReward(&rewards_);
    if(visualizable_ && !mode_){
      Eigen::Vector2d pos = controllerO_oppo.getTargetPos();
      targetSphere->setPosition(pos(0), pos(1), 0.1);
    }
    return exp(rewards_.sum());
  }

  void observe(Eigen::Ref<EigenVec> ob) {
    controller_.updateObservation(&world_);
    ob = controller_.getObservation().cast<float>();
  }

  void observe2(Eigen::Ref<EigenVec> obO) {
    if(mode_){
      controllerO_self.updateObservation(&world_);
      obO = controllerO_self.getObservation().cast<float>();
    }
    else{
      controllerO_oppo.setOppoState(gc_, gv_);
      controllerO_oppo.updateObservation(&world_);
      obO = controllerO_oppo.getObservation().cast<float>();
    }
  }

  int player1_die() {
    /// base contact with ground
    for(auto& contact: robot_->getContacts()) {
      if(contact.getPairObjectIndex() == world_.getObject("ground")->getIndexInWorld() &&
         contact.getlocalBodyIndex()  == robot_->getBodyIdx("base")) {
        return 1;
      }
    }
    /// get out of the cage
    if (gc_.head(2).norm() > 3) {
      return 2;
    }
    return 0;
  }

  int player2_die() {
    auto anymal = reinterpret_cast<raisim::ArticulatedSystem *>(world_.getObject(OPPONENT_NAME));
    /// base contact with ground
    for(auto& contact: anymal->getContacts()) {
      if(contact.getPairObjectIndex() == world_.getObject("ground")->getIndexInWorld() &&
          contact.getlocalBodyIndex() == anymal->getBodyIdx("base")) {
        return 1;
      }
    }
    if (gcOppo_.head(2).norm() > 3) {
      return 2;
    }
    return 0;
  }

  int object_die() {
    if (gcOppo_.head(2).norm() > 3) {
      return 2;
    }
    return 0;
  }

  void printTerminalInfo(const char* terminalType){
    std::string objName;
    if(objIdx < 0){ // flat
      if(objIdx == -100){
        objName = ", robot(pd)";
      }
      else{
        objName = ", robot";
      }
    }
    else if(objIdx == 0){
      objName = ", shpere";
    }
    else if(objIdx == 1){ 
      objName = "  box";
    }
    else if(objIdx == 2){
      objName = ",cylinder";
    }
    else if(objIdx == 3){
      objName = ", capsule";
    }
    else{
      objName = ",    error";
    }
    std::cout << terminalType << objName.c_str() << ", step = " << timer_ << std::endl;
  }

  bool isTerminalState(float &terminalReward) {
    robot_->getState(gc_, gv_);
    controller_.getOppoState(gcOppo_, gvOppo_);
    int dieInfo  = player1_die();
    int dieInfoO;
    std::string dieReason, dieReasonO;
    dieInfo = player1_die();
    if      (dieInfo  == 1) dieReason = "contact";
    else if (dieInfo  == 2) dieReason = "out    ";
    if (objIdx <  0) {
        dieInfoO = player2_die();
        if      (dieInfoO == 1) dieReasonO = "contact";
        else if (dieInfoO == 2) dieReasonO = "out    ";
    }
    else {
        dieInfoO = object_die();
        if      (dieInfoO == 1) dieReasonO = "contact";
        else if (dieInfoO == 2) dieReasonO = "out    ";
    }
    std::string message;
    if (dieInfo && dieInfoO) {
      draw += 1;
      message = "dr(" + dieReason + ", " + dieReasonO + ")";
      printTerminalInfo(message.c_str());
      terminalReward = -10;
      return true;
    }

    double rDiff = gc_.head(2).norm() - gcOppo_.head(2).norm() + 0.05;
    if (timer_ > 10 * 100) {
      draw += 1;
      if(rDiff < 0){
        printTerminalInfo("time over(    r win)");
        terminalReward = -5 - 2*rDiff;
      }
      else{
        printTerminalInfo("time over(   r lose)");
        terminalReward = -5 - 10*rDiff;
      }
      return true;
    }

    if (!dieInfo && dieInfoO) {
      player1_win += 1;
      message = "player1 win(" + dieReasonO + ")";
      printTerminalInfo(message.c_str());
      if(objIdx < 0){ // robot
        if((gc_.head(2) - gcOppo_.head(2)).norm() < 0.55){
          if(timer_ > 200) terminalReward = 12 - 2*rDiff;
          else terminalReward = 2 - 2*rDiff;
        }
        else terminalReward = -2 - 2*rDiff;
      }
      else{ // object
        if((gc_.head(2) - gcOppo_.head(2)).norm() < 1) terminalReward = 10.0;
        else terminalReward = -2 - 2*rDiff;
      }
      return true;
    }

    if (dieInfo && !dieInfoO) {
      player2_win += 1;
      message = "player2 win(" + dieReason + ")";
      printTerminalInfo(message.c_str());
      terminalReward = -150.0 - 10*rDiff;;
      return true;
    }
    return false;
  }


  void curriculumUpdate(){
  };

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
  raisim::ArticulatedSystem *robot_, *robotO_;
  AnymalController_20190430 controller_, controllerO_self;
  AnymalController_20002000 controllerO_oppo;
  bool mode_ = false;
  raisim::World world_;
  raisim::Reward rewards_;
  double simulation_dt_ = 0.001;
  double control_dt_ = 0.01;
  Eigen::VectorXd gc_, gv_;
  Eigen::VectorXd gcOppo_, gvOppo_;
  Eigen::Vector2d targetPos_;

  double factor_;
  // opponent object
  std::vector<raisim::SingleBodyObject*> objects_;
  raisim::Visuals* targetSphere;
  double minObjSize;
  int objIdx;
  int strategy;
  Eigen::Vector3d randomFactor_;


  std::unique_ptr<raisim::RaisimServer> server_;
  thread_local static std::normal_distribution<double> normDist_;
  thread_local static std::uniform_real_distribution<double> uniDist_;
  thread_local static std::mt19937 gen_;
};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
thread_local std::normal_distribution<double> raisim::ENVIRONMENT::normDist_(0., 1.);
thread_local std::uniform_real_distribution<double> raisim::ENVIRONMENT::uniDist_(0., 1.);
}
