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
#include <deque>
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

namespace raisim {

class ENVIRONMENT {

 public:

  explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable) :
      visualizable_(visualizable) {
    /// add objects
    robot_ = world_.addArticulatedSystem(resourceDir + "/anymal/urdf/anymal_red.urdf");
    robot_->setName(PLAYER_NAME);
    controller_.setName(PLAYER_NAME);
    robot_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

    robotO_ = world_.addArticulatedSystem(resourceDir + "/anymal/urdf/anymal_blue.urdf");
    robotO_->setName(OPPONENT_NAME);
    controllerO_.setName(OPPONENT_NAME);
    robotO_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

    auto* ground = world_.addGround();
    ground->setName("ground");

    controller_.create(&world_);
    controllerO_.create(&world_);
    READ_YAML(double, simulation_dt_, cfg["simulation_dt"])
    READ_YAML(double, control_dt_, cfg["control_dt"])

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

    /// visualize if it is the first environment
    if (visualizable_) {
      const int port = 8080;
      server_ = std::make_unique<raisim::RaisimServer>(&world_);
      server_->launchServer(port);
      server_->focusOn(robot_);
      auto cage = server_->addVisualCylinder("cage", 3.0, 0.05);
      cage->setPosition(0,0,0);
    }
    
    // add opponent
    objects_.push_back(world_.addSphere(0.1, 1.0));
    objects_.push_back(world_.addBox(0.1, 0.1, 0.1, 1.0));
    objects_.push_back(world_.addCylinder(0.1, 0.1, 1.0));
    objects_.push_back(world_.addCapsule(0.1, 0.1, 1.0));
    objIdx = 0;
    strategy = 0;
    discountFactorForSatefy_ = 1.0;
    randomFactor_.setZero();
    gc_.setZero(robot_->getGeneralizedCoordinateDim());
    gv_.setZero(robot_->getDOF() + 6);
    gcOppo_.setZero(robotO_->getGeneralizedCoordinateDim());
    gvOppo_.setZero(robotO_->getDOF() + 6);
    gcOppoBias_.setZero();
    rot_.setIdentity();
    rotO_.setIdentity();
    contactInfo_.setZero(15);
    contactInfoOppo_.setZero(15);
    contactFrames_.push_back(robot_->getFrameIdxByName("LF_KFE"));
    contactFrames_.push_back(robot_->getFrameIdxByName("RF_KFE"));
    contactFrames_.push_back(robot_->getFrameIdxByName("LH_KFE"));
    contactFrames_.push_back(robot_->getFrameIdxByName("RH_KFE"));
    contactFrames_.push_back(robot_->getFrameIdxByName("LF_HFE"));
    contactFrames_.push_back(robot_->getFrameIdxByName("RF_HFE"));
    contactFrames_.push_back(robot_->getFrameIdxByName("LH_HFE"));
    contactFrames_.push_back(robot_->getFrameIdxByName("RH_HFE"));
    contactFrames_.push_back(robot_->getFrameIdxByName("LF_HAA"));
    contactFrames_.push_back(robot_->getFrameIdxByName("RF_HAA"));
    contactFrames_.push_back(robot_->getFrameIdxByName("LH_HAA"));
    contactFrames_.push_back(robot_->getFrameIdxByName("RH_HAA"));
    
    contactIndices_.push_back(robot_->getBodyIdx("base"));
    contactIndices_.push_back(robot_->getBodyIdx("LF_SHANK"));
    contactIndices_.push_back(robot_->getBodyIdx("RF_SHANK"));
    contactIndices_.push_back(robot_->getBodyIdx("LH_SHANK"));
    contactIndices_.push_back(robot_->getBodyIdx("RH_SHANK"));
    contactIndices_.push_back(robot_->getBodyIdx("LF_THIGH"));
    contactIndices_.push_back(robot_->getBodyIdx("RF_THIGH"));
    contactIndices_.push_back(robot_->getBodyIdx("LH_THIGH"));
    contactIndices_.push_back(robot_->getBodyIdx("RH_THIGH"));
    contactIndices_.push_back(robot_->getBodyIdx("LF_HIP"));
    contactIndices_.push_back(robot_->getBodyIdx("RF_HIP"));
    contactIndices_.push_back(robot_->getBodyIdx("LH_HIP"));
    contactIndices_.push_back(robot_->getBodyIdx("RH_HIP"));
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
    discountFactorForSatefy_ = 1.0;
    int playerNum = 1*(uniDist_(gen_) > 0.0);
    controller_.setPlayerNum(playerNum);
    double theta = uniDist_(gen_) * 2 * M_PI;
    controller_.reset(&world_, theta);

    // set object pos
    Eigen::Vector3d objPos = Eigen::Vector3d(gc_(0), gc_(1), 1.2); 
    const Eigen::Vector4d objOri = randomQuatGen();
    while((objPos.head(2) - gc_.head(2)).norm() < 1.5){
      const double r = 1.35*(1 + uniDist_(gen_));
      const double theta = 2*M_PI*uniDist_(gen_);
      objPos << r*cos(theta), r*sin(theta), 1.2;
    }

    //
    strategy = int(uniDist_(gen_) > 0); // 0 is fight, 1 is go to the center
    randomFactor_ << 0.75 + 0.25*uniDist_(gen_), 0.8*normDist_(gen_), uniDist_(gen_); // force mag, direction, disturbance

    // reset states
    robot_->getState(gc_, gv_);
    rot_ = controller_.getRot();
    xyBefore_ << gc_.head(2), objPos;
    RR_ << gc_.head(2).norm(), objPos.head(2).norm(), 0, 0;
    gcOppo_.setZero();
    gvOppo_.setZero();
    gcOppo_.head(3) << objPos;
    const Eigen::Vector2d perturbedGcOppo = gcOppo_.head(2) + gcOppoBias_ +  0.1*Eigen::Vector2d(normDist_(gen_), normDist_(gen_));
    const Eigen::Vector3d perturbedvec = Eigen::Vector3d(perturbedGcOppo(0) - gc_(0), perturbedGcOppo(1) - gc_(1), 0);
    const double yaw = atan2(perturbedvec(1), perturbedvec(0)) - atan2(rot_(1,0), rot_(0,0)); // my yaw - me to opponent yaw
    Eigen::Matrix3d projectedRot;
    projectedRot << cos(yaw), -sin(yaw), 0,
                    sin(yaw), cos(yaw), 0,
                    0, 0, 1;
    Eigen::VectorXd state(57);
    state << // my info(37)
             gc_.head(3), /// body pose
             3.0 - gc_.head(2).norm(), /// distance to the edge
             rot_.row(2).transpose(), /// body orientation
             gc_.tail(12), /// joint angles
             rot_.transpose() * gv_.head(3), rot_.transpose() * gv_.segment(3, 3), /// body linear&angular velocity
             gv_.tail(12), /// joint velocity
             // opponent(5)
             perturbedGcOppo, /// opponent position
             (projectedRot*perturbedvec).head(2), /// opponent position relative to me
             3.0 - perturbedGcOppo.norm(), /// distance to the edge
             contactInfo_; // contact info
    myHistory_.clear();
    opponentHistory_.clear();
    for(int i = 0; i < 10; i++){
      myHistory_.push_front(state);
      opponentHistory_.push_front(gcOppo_.head(3).cast<float>());
    }
    actionHistory_.clear();
    Eigen::VectorXf action(12); action.setZero();
    for(int i = 0; i < 3; i++){
      actionHistory_.push_front(action);
    }
  }

  void desiredReward(){
    float actionSmooth = (actionHistory_[0] - actionHistory_[1]).squaredNorm()*(timer_ > 1);
    actionSmooth += (actionHistory_[0] - 2 * actionHistory_[1] + actionHistory_[2]).squaredNorm()*(timer_ > 2);
    rewards_.record("f_smooth", double(actionSmooth));
  }

  float step(const Eigen::Ref<EigenVec> &action) {
    actionHistory_.push_front(action);
    actionHistory_.pop_back();
    timer_++;
    controller_.advance(&world_, action);

    // robot_->setExternalForce(0, -externalForce);
    for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
      if (server_) server_->lockVisualizationServerMutex();
      world_.integrate();
      if (server_) server_->unlockVisualizationServerMutex();
    }
    // update states
    robot_->getState(gc_, gv_);

    // calculate reward
    desiredReward();
    // how much move along to the (past gc to current opponent) line
    Eigen::Vector2d pastToOppo = (gcOppo_.head(2) - myHistory_[5].head(2)).normalized();
    Eigen::Vector2d pastToCur  = gc_.head(2) - myHistory_[5].head(2);
    double relative = pastToCur.dot(pastToOppo);
    rewards_.record("e_relative", relative*(1 + (relative > 0)*(discountFactorForSatefy_ - 1)));
    rewards_.record("g_survive", 0);

    return exp(rewards_.sum());
  }


  void observe(Eigen::Ref<EigenVec> ob) {
    robot_->getState(gc_, gv_);
    rot_ = controller_.getRot();
    contactInfo_.setZero();
    controller_.updateObservation(&world_);

    if(objIdx < 0){
      robotO_->getState(gcOppo_, gvOppo_);
    }
    else{
      raisim::Vec<3> oppoPos, oppoVel;
      objects_[objIdx]->getPosition(0, oppoPos);
      objects_[objIdx]->getVelocity(0, oppoVel);
      gcOppo_.head(3) = oppoPos.e();
      gvOppo_.head(3) = oppoVel.e();
    }
    
    const Eigen::Vector2d perturbedGcOppo = gcOppo_.head(2) + gcOppoBias_ + (0.2 + 0.8*(objIdx > 0))*0.1*Eigen::Vector2d(normDist_(gen_), normDist_(gen_));
    const Eigen::Vector3d perturbedvec = Eigen::Vector3d(perturbedGcOppo(0) - gc_(0), perturbedGcOppo(1) - gc_(1), 0);
    const double yaw = atan2(perturbedvec(1), perturbedvec(0)) - atan2(rot_(1,0), rot_(0,0)); // my yaw - me to opponent yaw
    Eigen::Matrix3d projectedRot;
    projectedRot << cos(yaw), -sin(yaw), 0,
                    sin(yaw), cos(yaw), 0,
                    0, 0, 1;
    Eigen::VectorXd state(57);
    state << // my info(37)
             gc_.head(3), /// body pose
             3.0 - gc_.head(2).norm(), /// distance to the edge
             rot_.row(2).transpose(), /// body orientation
             gc_.tail(12), /// joint angles
             rot_.transpose() * gv_.head(3), rot_.transpose() * gv_.segment(3, 3), /// body linear&angular velocity
             gv_.tail(12), /// joint velocity
             // opponent(5)
             perturbedGcOppo, /// opponent position
             (projectedRot*perturbedvec).head(2), /// opponent position relative to me
             3 - gcOppo_.head(2).norm(), /// distance to the edge
             contactInfo_; // contact info

    ob << state.cast<float>(), myHistory_[1].cast<float>(), myHistory_[2].cast<float>();

    // queue, pop
    myHistory_.push_front(state);
    myHistory_.pop_back();
    // opponentHistory_.push_front(gcOppo_.head(3).cast<float>());
    // opponentHistory_.pop_back();
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

  int object_die() {
    if (objects_[objIdx]->getPosition().head(2).norm() > 3) {
      return 2;
    }
    return 0;
  }

  void printTerminalInfo(const char* terminalType){
    std::cout << terminalType << ", step = " << timer_ << std::endl;
  }

  bool isTerminalState(float &terminalReward) {
    int dieInfo  = player1_die();
    int dieInfoO;
    std::string dieReason, dieReasonO;
    dieInfo = player1_die();
    if      (dieInfo  == 1) dieReason = "contact";
    else if (dieInfo  == 2) dieReason = "out    ";

    std::string message;
    if (dieInfo) {
      player2_win += 1;
      message = "terminated(" + dieReason + ")";
      printTerminalInfo(message.c_str());
      terminalReward = -100.0;
      return true;
    }
    return false;
  }

  void curriculumUpdate() {};

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
  AnymalController_20190430 controller_, controllerO_;
  raisim::World world_;
  raisim::Reward rewards_;
  double simulation_dt_ = 0.001;
  double control_dt_ = 0.01;
  Eigen::VectorXd gc_, gv_;
  Eigen::VectorXd gcOppo_, gvOppo_;
  Eigen::Vector2d gcOppoBias_;
  Eigen::Vector4d xyBefore_, RR_;
  Eigen::Matrix3d rot_, rotO_;
  std::vector<size_t> contactIndices_, contactFrames_;
  Eigen::VectorXd contactInfo_, contactInfoOppo_;
  std::deque<Eigen::VectorXf> opponentHistory_;
  std::deque<Eigen::VectorXf> actionHistory_, actionHistoryO_;
  std::deque<Eigen::VectorXd> myHistory_;
  double discountFactorForSatefy_;

  // opponent object
  std::vector<raisim::SingleBodyObject*> objects_;
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
