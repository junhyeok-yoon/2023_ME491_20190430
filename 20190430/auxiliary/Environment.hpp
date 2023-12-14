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

    // Set object, size, oreientation, properties
    std::uniform_int_distribution<int> objDist_(-5,3);
    objIdx = objDist_(gen_);
    if(objIdx >= 0) objIdx = objDist_(gen_);
    const double mass = 35.0 + 15.0*uniDist_(gen_);
    double mu = 0.7 + 0.4*uniDist_(gen_);
    double height;
    if(objIdx < 0){ // robot
      gcOppoBias_.setZero();
      if(objIdx < -2){
        // randomize gc init
        Eigen::VectorXd gc_init_tail; gc_init_tail.setZero(12);
        gc_init_tail << 0.2, 0.4, -0.75, -0.2, 0.4, -0.75, 0.2, -0.4, 0.75, -0.2, -0.4, 0.75;
        for(int i = 0; i < 4; i++){
          gc_init_tail(3*i    ) += 0.15*normDist_(gen_);
          gc_init_tail(3*i + 1) += 0.15*normDist_(gen_);
          gc_init_tail(3*i + 2) += 0.35*normDist_(gen_);
        }
        controllerO_.resetGcInit(gc_init_tail);
      }
      objPos(2) = 0.6 + 0.15*uniDist_(gen_);
      controllerO_.setPlayerNum(1 - playerNum);
      controllerO_.resetGivenPos(&world_, objPos);
      robotO_->getState(gcOppo_, gvOppo_);
      minObjSize = 0.15;
    }
    else{ // object
      // remove previous object
      if(objects_[objIdx] != nullptr) {
        world_.removeObject(objects_[objIdx]);
      }
      // introduce bias in CoM
      gcOppoBias_ << 0.1*normDist_(gen_), 0.1*normDist_(gen_);
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
      objects_[objIdx]->setPosition(objPos(0), objPos(1), objPos(2));
      objects_[objIdx]->setOrientation(objOri(0), objOri(1), objOri(2), objOri(3));
      controllerO_.resetFar(&world_);
    }
    for(int i = 0; i < 4; i++) if(i != objIdx) objects_[i]->setPosition(30, 30, -100);

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

  void reset2() {
    timer_ = 0;
    const int playerNum = 1*(uniDist_(gen_) > 0.0);
    const int playerNumOppo = 1 - playerNum;
    const double theta = uniDist_(gen_) * 2 * M_PI;
    controller_.setPlayerNum(playerNum);
    controller_.reset(&world_, theta);
    controllerO_.setPlayerNum(playerNumOppo);
    controllerO_.reset(&world_, theta);
    objIdx = -1;
    // reset states
    robot_->getState(gc_, gv_);
    robotO_->getState(gcOppo_, gvOppo_);
    rot_ = controller_.getRot();
    rotO_ = controllerO_.getRot();
    xyBefore_ << gc_.head(2), gcOppo_.head(2);
    RR_ << gc_.head(2).norm(), gcOppo_.head(2).norm(), 0, 0;

    for(int i = 0; i < 4; i++) objects_[i]->setPosition(30, 30, -100);

    double yaw = atan2(gcOppo_(1) - gc_(1), gcOppo_(0) - gc_(0)) - atan2(rot_(1,0), rot_(0,0)); // my yaw - me to opponent yaw
    Eigen::Matrix3d projectedRot;
    projectedRot << cos(yaw), -sin(yaw), 0,
                    sin(yaw), cos(yaw), 0,
                    0, 0, 1;
    const Eigen::Vector3d ab = Eigen::Vector3d(gcOppo_(1) - gc_(1), gcOppo_(0) - gc_(0), 0);
    Eigen::VectorXd state(57);
    state << // my info(37)
             gc_.head(3), /// body pose
             3.0 - gc_.head(2).norm(), /// distance to the edge
             rot_.row(2).transpose(), /// body orientation
             gc_.tail(12), /// joint angles
             rot_.transpose() * gv_.head(3), rot_.transpose() * gv_.segment(3, 3), /// body linear&angular velocity
             gv_.tail(12), /// joint velocity
             // opponent(5)
             gcOppo_.head(2), /// opponent position
             (projectedRot*ab).head(2), /// opponent position relative to me
             3.0 - gcOppo_.head(2).norm(), /// distance to the edge
             contactInfo_;
    
    yaw = atan2(gc_(1) - gcOppo_(1), gc_(0) - gcOppo_(0)) - atan2(rotO_(1,0), rotO_(0,0)); // my yaw - me to opponent yaw
    projectedRot << cos(yaw), -sin(yaw), 0,
                    sin(yaw), cos(yaw), 0,
                    0, 0, 1;
    Eigen::VectorXd oppoState(42);
    oppoState << // oppo info(37)
                 gcOppo_.head(3), /// body pose
                 3.0 - gcOppo_.head(2).norm(), /// distance to the edge
                 rotO_.row(2).transpose(), /// body orientation
                 gcOppo_.tail(12), /// joint angles
                 rotO_.transpose() * gvOppo_.head(3), rotO_.transpose() * gvOppo_.segment(3, 3), /// body linear&angular velocity
                 gvOppo_.tail(12), /// joint velocity
                 // me(5)
                 gc_.head(2), /// opponent position
                 -(projectedRot*ab).head(2), /// opponent position relative to me
                 3.0 - gc_.head(2).norm(); /// distance to the edge

    myHistory_.clear();
    opponentHistory_.clear();
    for(int i = 0; i < 10; i++){
      myHistory_.push_front(state);
      opponentHistory_.push_front(oppoState.cast<float>());
    }
    actionHistory_.clear();
    actionHistoryO_.clear();
    Eigen::VectorXf action(12); action.setZero();
    for(int i = 0; i < 3; i++){
      actionHistory_.push_front(action);
      actionHistoryO_.push_front(action);
    }
  }

  void fightReward(){
    bool opponentContact = false;
    Eigen::Vector3d contactForce;
    int contactCount = 0;
    for(auto& contact: robot_->getContacts()) {
      if(objIdx < 0){
        if(contact.getPairObjectIndex() == robotO_->getIndexInWorld()) {
          opponentContact = true;
        }
      }
      else{
        if(contact.getPairObjectIndex() == objects_[objIdx]->getIndexInWorld()) {
          opponentContact = true;
        }
      }
      const raisim::Vec<3> force = contact.getImpulse();
      contactForce += force.e();
      // std::cout << "force = " << force.e().transpose() << "contactCount = " << contactCount << std::endl;
      contactCount++;
    }
    double fightReward = 0;
    double rpyReward = 0;

    double rDiff = RR_(0) - RR_(1) + 0.1;
    if(rDiff > 0.0 && RR_(0) > 2.0){
      fightReward -= 0.05*rDiff*rDiff; 
    }
    if(opponentContact){ // contact with opponent or close to the opponent
      fightReward = 0.005; // contact reward(generally good)
      fightReward += 2.5*(gcOppo_.head(2).norm() - xyBefore_.segment(2,2).norm()); // move away from the center
      const Eigen::Vector2d me2oppo_before = (xyBefore_.segment(2,2) - xyBefore_.head(2)).normalized();
      const Eigen::Vector2d oppoDelta = gcOppo_.head(2) - xyBefore_.segment(2,2);
      const double myDelta = oppoDelta.dot(me2oppo_before); // my move along to the (past gc to past opponent) line
      if(myDelta < 0){
        fightReward += 2*myDelta;
      }
      else{
        fightReward += myDelta;
      }
      
      fightReward += oppoDelta.dot(me2oppo_before); // opponent's move along to the (past gc to past opponent) line
      // const Eigen::Vector2d myDelta = gc_.head(2) - xyBefore_.head(2);
      // fightReward -= myDelta.dot(me2oppo_before); // my move along to the (past gc to past opponent) line

      //
      if(objIdx < 0){
        if(gcOppo_.segment(3,4).isZero()){std::cout << "gcOppo_ quat is zerro" << std::endl;}
        else{
          Eigen::Vector2d rpyOppo = quatToRPY(gcOppo_.segment(3,4)).head(2);
          rpyOppo.cwiseAbs();
          rpyOppo(0) = std::min(std::max(rpyOppo(0) - 0.2, 0.0), 0.8);
          rpyOppo(1) = std::min(std::max(rpyOppo(1) - 0.2, 0.0), 0.8);
          Eigen::Vector3d velOppo = gcOppo_.head(3);
          velOppo(0) = std::min(std::max(velOppo(0) - 0.05, 0.0), 0.8);
          velOppo(1) = std::min(std::max(velOppo(1) - 0.05, 0.0), 0.8);
          velOppo(2) = std::min(std::max(velOppo(2) - 0.05, 0.0), 0.8);
          Eigen::Vector2d meToOppo = gcOppo_.head(2) - gc_.head(2);
          meToOppo = meToOppo.normalized();
          double projectedVel = velOppo.head(2).dot(meToOppo);
          rpyReward += rpyOppo.squaredNorm() + 2*pow(projectedVel*(projectedVel > 0), 2) + 3 * velOppo(2)*velOppo(2);
        }
      }
    }
    Eigen::Vector2d rpyMy = quatToRPY(gc_.segment(3,4)).head(2);
    rpyMy.cwiseAbs();
    rpyMy(0) = std::min(std::max(rpyMy(0) - 0.3, 0.0), 1.2);
    rpyMy(1) = std::min(std::max(rpyMy(1) - 0.3, 0.0), 1.2);
    rpyReward -= 3*rpyMy.squaredNorm();

    rewards_.record("d_fight", fightReward*(1 + (fightReward > 0)*(discountFactorForSatefy_-1)));
    rewards_.record("i_rpy", rpyReward);
  }

  void desiredReward(){
    float actionSmooth = (actionHistory_[0] - actionHistory_[1]).squaredNorm()*(timer_ > 1);
    actionSmooth += (actionHistory_[0] - 2 * actionHistory_[1] + actionHistory_[2]).squaredNorm()*(timer_ > 2);
    rewards_.record("f_smooth", double(actionSmooth));
  }

  float step(const Eigen::Ref<EigenVec> &action) {
    if(objIdx < 0){
      Eigen::VectorXf oppoAction(12); oppoAction.setZero();
      oppoAction.setZero();
      return step2(action, oppoAction);
    }
    actionHistory_.push_front(action);
    actionHistory_.pop_back();
    timer_++;
    controller_.advance(&world_, action);
    raisim::Vec<3> oppoPos, oppoVel;
    objects_[objIdx]->getPosition(0, oppoPos);
    objects_[objIdx]->getVelocity(0, oppoVel);
    gcOppo_.head(3) = oppoPos.e();
    gvOppo_.head(3) = oppoVel.e();

    // external force on the object
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
    // update states
    robot_->getState(gc_, gv_);
    objects_[objIdx]->getPosition(0, oppoPos);
    objects_[objIdx]->getVelocity(0, oppoVel);
    gcOppo_.head(3) = oppoPos.e();
    gvOppo_.head(3) = oppoVel.e();


    // calculate reward
    RR_ << gc_.head(2).norm(), gcOppo_.head(2).norm(), 0, 0;
    controller_.recordReward(&rewards_);
    double safety = std::pow(std::max(RR_(0) - 1.0, 0.0), 2);
    safety += 20*std::pow(std::max(RR_(0) - 2.0, 0.0), 4);
    discountFactorForSatefy_ = 1 - (RR_(0) > 2.5)*(RR_(0) - 2.5)/0.5; 
    rewards_.record("c_safety", safety);
    fightReward();
    desiredReward();
    // how much move along to the (past gc to current opponent) line
    Eigen::Vector2d pastToOppo = (gcOppo_.head(2) - myHistory_[5].head(2)).normalized();
    Eigen::Vector2d pastToCur  = gc_.head(2) - myHistory_[5].head(2);
    double relative = pastToCur.dot(pastToOppo);
    rewards_.record("e_relative", relative*(1 + (relative > 0)*(discountFactorForSatefy_ - 1)));
    rewards_.record("g_survive", 0);
    const double r = std::max((gc_.head(2) - gcOppo_.head(2)).norm() - minObjSize, 0.0);
    if(isnan(r)){
      std::cout << "gc_ = " << gc_.transpose() << std::endl;
      std::cout << "gcOppo_ = " << gcOppo_.transpose() << std::endl;
      std::cout << "minObjSize = " << minObjSize << std::endl;
    }
    rewards_.record("h_distance", square(r));
    // rewards_.record("c_safety", xyBefore_.head(2).norm()      - gc_.head(2).norm());
    // rewards_.record("d_fight",  xyBefore_.segment(2,2).norm() - objects_[objIdx]->getPosition().head(2).norm());
    // rewards_.record("e_relative", std::max((gc_.head(2) - objects_[objIdx]->getPosition().head(2)).norm() - minObjSize, 0.0));
    xyBefore_ << gc_.head(2), gcOppo_.head(2);
    return exp(rewards_.sum());
  }

  float step2(const Eigen::Ref<EigenVec> &action, const Eigen::Ref<EigenVec> &actionO) {
    actionHistory_.push_front(action);
    actionHistory_.pop_back();
    actionHistoryO_.push_front(actionO);
    actionHistoryO_.pop_back();
    timer_++;
    controller_.advance(&world_, action);
    if(objIdx != -5) controllerO_.advance(&world_, actionO);
    //
    for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
      if (server_) server_->lockVisualizationServerMutex();
      world_.integrate();
      if (server_) server_->unlockVisualizationServerMutex();
    }
    // update states
    robot_->getState(gc_, gv_);
    robotO_->getState(gcOppo_, gvOppo_);

    // calculate reward
    RR_ << gc_.head(2).norm(), gcOppo_.head(2).norm(), 0, 0;
    controller_.recordReward(&rewards_);
    double safety = std::pow(std::max(RR_(0) - 1.0, 0.0), 2);
    safety += 20*std::pow(std::max(RR_(0) - 2.0, 0.0), 4);
    discountFactorForSatefy_ = 1 - (RR_(0) > 2.5)*(RR_(0) - 2.5)/0.5; 
    rewards_.record("c_safety", safety);
    fightReward();
    desiredReward();
    // how much move along to the (past gc to current opponent) line
    Eigen::Vector2d pastToOppo = (gcOppo_.head(2) - myHistory_[5].head(2)).normalized();
    Eigen::Vector2d pastToCur  = gc_.head(2) - myHistory_[5].head(2);
    double relative = pastToCur.dot(pastToOppo);
    rewards_.record("e_relative", relative*(1 + (relative > 0)*(discountFactorForSatefy_ - 1)));
    rewards_.record("g_survive", 0);
    const double r = std::max((gc_.head(2) - gcOppo_.head(2)).norm() - minObjSize, 0.0);
    if(isnan(r)){
      std::cout << "step2 gc_ = " << gc_.transpose() << std::endl;
      std::cout << "gcOppo_ = " << gcOppo_.transpose() << std::endl;
      std::cout << "minObjSize = " << minObjSize << std::endl;
    }
    rewards_.record("h_distance", square(r));
    // rewards_.record("c_safety", xyBefore_.head(2).norm()      - gc_.head(2).norm());
    // rewards_.record("d_fight",  xyBefore_.segment(2,2).norm() - objects_[objIdx]->getPosition().head(2).norm());
    // rewards_.record("e_relative", std::max((gc_.head(2) - objects_[objIdx]->getPosition().head(2)).norm() - minObjSize, 0.0));
    xyBefore_ << gc_.head(2), gcOppo_.head(2);
    return exp(rewards_.sum());
  }

  void getContactInfo(){
    contactInfo_.setZero();
    for(auto& contact: robot_->getContacts()){
      if (contact.skip()) continue;
      for(int i = 0; i < 13; i++){
        if(contact.getPairObjectIndex() != world_.getObject("ground")->getIndexInWorld() && contactIndices_[i] == contact.getlocalBodyIndex()){
          if(i==0){
            contactInfo_.head(3) = rot_.transpose()*(contact.getPosition().e() - gc_.head(3));
          }
          else{
            raisim::Vec<3> framePos;
            robot_->getFramePosition(contactFrames_[i-1], framePos);
            contactInfo_(i + 2) = (contact.getPosition().e() - framePos.e()).norm();
          }
          // std::cout << "contact force = " << (contact.getContactFrame().e().transpose() * contact.getImpulse().e()).transpose() << std::endl;
          // std::cout << "contactInfo_(" << i + 2 << ") = " << contactInfo_(i + 2) << std::endl;  
          break;
        }
      }
    }
  }

  void getContactInfoOppo(){
    contactInfoOppo_.setZero();
    for(auto& contact: robotO_->getContacts()){
      if (contact.skip()) continue; // if contact points is not calves, terminate
      for(int i = 0; i < 13; i++){
        if(contact.getPairObjectIndex() != world_.getObject("ground")->getIndexInWorld() && contactIndices_[i] == contact.getlocalBodyIndex()){
          if(i==0){
            contactInfo_.head(3) = rotO_.transpose()*(contact.getPosition().e() - gcOppo_.head(3));
          }
          else{
            raisim::Vec<3> framePos;
            robotO_->getFramePosition(contactFrames_[i-1], framePos);
            contactInfo_(i + 2) = (contact.getPosition().e() - framePos.e()).norm();
          }
          break;
        }
      }
    }
  }

  void observe(Eigen::Ref<EigenVec> ob) {
    robot_->getState(gc_, gv_);
    rot_ = controller_.getRot();
    getContactInfo();
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

    ob << state.cast<float>(), myHistory_[1].cast<float>(), myHistory_[2].cast<float>(), actionHistory_[0].cast<float>();;

    // queue, pop
    myHistory_.push_front(state);
    myHistory_.pop_back();
    // opponentHistory_.push_front(gcOppo_.head(3).cast<float>());
    // opponentHistory_.pop_back();
  }

  void observe2(Eigen::Ref<EigenVec> ob, Eigen::Ref<EigenVec> obO) {
    robot_->getState(gc_, gv_);
    rot_ = controller_.getRot();
    robotO_->getState(gcOppo_, gvOppo_);
    rotO_ = controllerO_.getRot();
    getContactInfo();
    getContactInfoOppo();
    
    controller_.updateObservation(&world_);
    controllerO_.updateObservation(&world_);
    
    const double yaw = atan2(gcOppo_(1) - gc_(1), gcOppo_(0) - gc_(0)) - atan2(rot_(1,0), rot_(0,0)); // my yaw - me to opponent yaw
    Eigen::Matrix3d projectedRot;
    projectedRot << cos(yaw), -sin(yaw), 0,
                    sin(yaw), cos(yaw), 0,
                    0, 0, 1;
    const Eigen::Vector3d ab = Eigen::Vector3d(gcOppo_(1) - gc_(1), gcOppo_(0) - gc_(0), 0);
    // make observation vector
    Eigen::VectorXd state(57);
    state << // my info(37)
             gc_.head(3), /// body pose
             3.0 - gc_.head(2).norm(), /// distance to the edge
             rot_.row(2).transpose(), /// body orientation
             gc_.tail(12), /// joint angles
             rot_.transpose() * gv_.head(3), rot_.transpose() * gv_.segment(3, 3), /// body linear&angular velocity
             gv_.tail(12), /// joint velocity
             // opponent(5)
             gcOppo_.head(2), /// opponent position
             (projectedRot*ab).head(2), /// opponent position relative to me
             3 - gcOppo_.head(2).norm(), /// distance to the edge
             contactInfo_; // contact info

    ob << state.cast<float>(), myHistory_[1].cast<float>(), myHistory_[2].cast<float>();

    projectedRot << -cos(yaw), sin(yaw), 0,
                    -sin(yaw), -cos(yaw), 0,
                    0, 0, 1;
    Eigen::VectorXd oppoState(42);
    oppoState << // oppo info(37)
                 gcOppo_.head(3), /// body pose
                 3.0 - gcOppo_.head(2).norm(), /// distance to the edge
                 rotO_.row(2).transpose(), /// body orientation
                 gcOppo_.tail(12), /// joint angles
                 rotO_.transpose() * gvOppo_.head(3), rotO_.transpose() * gvOppo_.segment(3, 3), /// body linear&angular velocity
                 gvOppo_.tail(12), /// joint velocity
                 // me(5)
                 gc_.head(2), /// opponent position
                 -(projectedRot*ab).head(2), /// opponent position relative to me
                 3.0 - gc_.head(2).norm(); /// distance to the edge

    obO << oppoState.cast<float>(), opponentHistory_[1], opponentHistory_[2];

    // queue, pop
    myHistory_.push_front(state);
    myHistory_.pop_back();
    opponentHistory_.push_front(oppoState.cast<float>());
    opponentHistory_.pop_back();
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
    if (objects_[objIdx]->getPosition().head(2).norm() > 3) {
      return 2;
    }
    return 0;
  }

  void printTerminalInfo(const char* terminalType){
    std::string objName;
    if(objIdx < 0){ // flat
      objName = ",    robot";
      // objName = ", robot(" + std::to_string(factor_) + ")";
    }
    else if(objIdx == 0){
      objName = ",   shpere";
    }
    else if(objIdx == 1){ 
      objName = ",      box";
    }
    else if(objIdx == 2){
      objName = ", cylinder";
    }
    else if(objIdx == 3){
      objName = ",  capsule";
    }
    else{
      objName = ",    error";
    }
    std::cout << terminalType << objName.c_str() << ", step = " << timer_ << std::endl;
  }

  bool isTerminalState(float &terminalReward) {
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

    if (timer_ > 20 * 100) {
      draw += 1;
      printTerminalInfo("time over draw      ");
      terminalReward = -20;
      return true;
    }

    if (!dieInfo && dieInfoO) {
      player1_win += 1;
      message = "player1 win(" + dieReasonO + ")";
      printTerminalInfo(message.c_str());
      if(objIdx < 0){ // robot
        if((gc_.head(2) - gcOppo_.head(2)).norm() < 0.55){
          if(timer_ > 200) terminalReward = 3.0;
          else terminalReward = 0.0;
        }
        else terminalReward = -5;
      }
      else{ // object
        if((gc_.head(2) - gcOppo_.head(2)).norm() < 1) terminalReward = 3.0;
        else terminalReward = -5;
      }
      return true;
    }

    if (dieInfo && !dieInfoO) {
      player2_win += 1;
      message = "player2 win(" + dieReason + ")";
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
