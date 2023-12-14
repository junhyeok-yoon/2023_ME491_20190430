// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

#pragma once

#include <set>
#include <deque>
#include "../../BasicEigenTypes.hpp"
#include "raisim/World.hpp"

namespace raisim {

/// change the class name and file name ex) AnymalController_00000000 -> AnymalController_STUDENT_ID
class AnymalController_20190430 {

 public:
  inline bool create(raisim::World *world) {
    anymal_ = reinterpret_cast<raisim::ArticulatedSystem *>(world->getObject(name_));
    anymalO_ = reinterpret_cast<raisim::ArticulatedSystem *>(world->getObject(opponentName_));
    world_ = reinterpret_cast<raisim::World *>(world);
    /// get robot data
    gcDim_ = anymal_->getGeneralizedCoordinateDim();
    gvDim_ = anymal_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_);
    gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_);
    gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_);
    vTarget_.setZero(gvDim_);
    pTarget12_.setZero(nJoints_);

    gcOppo_.setZero(gcDim_);
    gvOppo_.setZero(gvDim_);
    gcOppoBias_.setZero();

    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.50, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero();
    jointPgain.tail(nJoints_).setConstant(50.0);
    jointDgain.setZero();
    jointDgain.tail(nJoints_).setConstant(0.2);
    anymal_->setPdGains(jointPgain, jointDgain);
    anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 3*57 + 12;
    actionDim_ = nJoints_;
    actionMean_.setZero(actionDim_);
    actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(0.1);

    //
    timer_ = 0;
    pdStandingMode_ = false;
    xyBefore_.setZero();
    RR_.setZero();
    contactInfo_.setZero(15);
    discountFactorForSatefy_ = 1.0;

    /// indices of links that should not make contact with ground
    footIndices_.insert(anymal_->getBodyIdx("LF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("LH_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RH_SHANK"));


    contactFrames_.push_back(anymal_->getFrameIdxByName("LF_KFE"));
    contactFrames_.push_back(anymal_->getFrameIdxByName("RF_KFE"));
    contactFrames_.push_back(anymal_->getFrameIdxByName("LH_KFE"));
    contactFrames_.push_back(anymal_->getFrameIdxByName("RH_KFE"));
    contactFrames_.push_back(anymal_->getFrameIdxByName("LF_HFE"));
    contactFrames_.push_back(anymal_->getFrameIdxByName("RF_HFE"));
    contactFrames_.push_back(anymal_->getFrameIdxByName("LH_HFE"));
    contactFrames_.push_back(anymal_->getFrameIdxByName("RH_HFE"));
    contactFrames_.push_back(anymal_->getFrameIdxByName("LF_HAA"));
    contactFrames_.push_back(anymal_->getFrameIdxByName("RF_HAA"));
    contactFrames_.push_back(anymal_->getFrameIdxByName("LH_HAA"));
    contactFrames_.push_back(anymal_->getFrameIdxByName("RH_HAA"));
    
    contactIndices_.push_back(anymal_->getBodyIdx("base"));
    contactIndices_.push_back(anymal_->getBodyIdx("LF_SHANK"));
    contactIndices_.push_back(anymal_->getBodyIdx("RF_SHANK"));
    contactIndices_.push_back(anymal_->getBodyIdx("LH_SHANK"));
    contactIndices_.push_back(anymal_->getBodyIdx("RH_SHANK"));
    contactIndices_.push_back(anymal_->getBodyIdx("LF_THIGH"));
    contactIndices_.push_back(anymal_->getBodyIdx("RF_THIGH"));
    contactIndices_.push_back(anymal_->getBodyIdx("LH_THIGH"));
    contactIndices_.push_back(anymal_->getBodyIdx("RH_THIGH"));
    contactIndices_.push_back(anymal_->getBodyIdx("LF_HIP"));
    contactIndices_.push_back(anymal_->getBodyIdx("RF_HIP"));
    contactIndices_.push_back(anymal_->getBodyIdx("LH_HIP"));
    contactIndices_.push_back(anymal_->getBodyIdx("RH_HIP"));


    return true;
  }

  inline bool init(raisim::World *world) {
    return true;
  }

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

  inline bool advance(raisim::World *world, const Eigen::Ref<EigenVec> &action) {
    timer_++;
    if(pdStandingMode_){
      pTarget_.tail(nJoints_) = gc_init_.tail(nJoints_);
      anymal_->setPdTarget(pTarget_, vTarget_);
      return true;
    }
    actionHistory_.push_front(action);
    actionHistory_.pop_back();
    /// action scaling
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;
    anymal_->setPdTarget(pTarget_, vTarget_);
    return true;
  }

  bool resetGivenPos(raisim::World *world, Eigen::Vector3d pos, Eigen::Vector3d oppoPos) {
    timer_ = 0;
    objIdx_ = -1;
    pdStandingMode_ = false;
    gc_init_.head(3) << pos;
    anymal_->setState(gc_init_, gv_init_);
    raisim::Vec<4> quat;
    raisim::Mat<3, 3> rot;
    quat[0] = gc_init_[3];
    quat[1] = gc_init_[4];
    quat[2] = gc_init_[5];
    quat[3] = gc_init_[6];
    raisim::quatToRotMat(quat, rot);
    rot_ = rot.e();
    contactInfo_.setZero();

    // reset states
    anymal_->getState(gc_, gv_);
    gcOppo_.head(3) << oppoPos;
    gvOppo_.head(3).setZero();
    Eigen::VectorXd state(57); state.setZero();
    myHistory_.clear();
    opponentHistory_.clear();
    Eigen::VectorXf oppoState(6); oppoState.setZero();
    for(int i = 0; i < 10; i++){
      myHistory_.push_front(state);
      opponentHistory_.push_front(oppoState);
    }
    actionHistory_.clear();
    Eigen::VectorXf action(12); action.setZero();
    for(int i = 0; i < 3; i++){
      actionHistory_.push_front(action);
    }

    return true;
  }

  void resetGcInit(Eigen::VectorXd& gc_init_tail){
    gc_init_.tail(nJoints_) = gc_init_tail;
    actionMean_ = gc_init_.tail(nJoints_);
  }

  void resetFar(raisim::World *world) {
    gc_init_.head(3) << 30, 30, -100;
    anymal_->setState(gc_init_, gv_init_);
  }

  inline bool reset(raisim::World *world, double theta) {
    timer_ = 0;
    objIdx_ = -1;
    pdStandingMode_ = false;
    if (playerNum_ == 0) {
      gc_init_.head(3) << 1.5 * std::cos(theta), 1.5 * std::sin(theta), 0.5;
      gc_init_.segment(3, 4) << cos((theta - M_PI) / 2), 0, 0, sin((theta - M_PI) / 2);
    }
    else {
      gc_init_.head(3) << 1.5 * std::cos(theta + M_PI), 1.5 * std::sin(theta + M_PI), 0.5;
      gc_init_.segment(3, 4) << cos(theta / 2), 0, 0, sin(theta / 2);
    }
    anymal_->setState(gc_init_, gv_init_);
    anymal_->getState(gc_, gv_);
    anymalO_->getState(gcOppo_, gvOppo_);
    gcOppoBias_.setZero();
    
    raisim::Vec<4> quat;
    raisim::Mat<3, 3> rot;
    quat[0] = gc_init_[3];
    quat[1] = gc_init_[4];
    quat[2] = gc_init_[5];
    quat[3] = gc_init_[6];
    raisim::quatToRotMat(quat, rot);
    rot_ = rot.e();
    contactInfo_.setZero();

    // reset states
    anymal_->getState(gc_, gv_);
    xyBefore_ << gc_.head(2), gcOppo_.head(2);
    RR_ << gc_.head(2).norm(), gcOppo_.head(2).head(2).norm(), 0, 0;
    discountFactorForSatefy_ = 1.0;

    const Eigen::Vector2d perturbedGcOppo = gcOppo_.head(2);// + gcOppoBias_ +  0.1*Eigen::Vector2d(normDist_(gen_), normDist_(gen_));
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
    Eigen::VectorXf oppoState(6); 
    oppoState << gcOppo_.head(3).cast<float>(), gvOppo_.head(3).cast<float>();
    for(int i = 0; i < 10; i++){
      myHistory_.push_front(state);
      opponentHistory_.push_front(oppoState);
    }
    actionHistory_.clear();
    Eigen::VectorXf action(12); action.setZero();
    for(int i = 0; i < 3; i++) actionHistory_.push_front(action);

    return true;
  }

  bool reset2(raisim::World *world, double theta, int objIdx = -1, raisim::SingleBodyObject* objPtr = NULL) {
    timer_ = 0;
    objIdx_ = objIdx;
    pdStandingMode_ = false;
    if(objIdx <= -2){
      if(objIdx == -100) pdStandingMode_ = true;
      // randomize gc init
      gc_init_.tail(12) << 0.2, 0.4, -0.75, -0.2, 0.4, -0.75, 0.2, -0.4, 0.75, -0.2, -0.4, 0.75;
      for(int i = 0; i < 4; i++){
        gc_init_(3*i + 7) += 0.15*normDist_(gen_);
        gc_init_(3*i + 8) += 0.15*normDist_(gen_);
        gc_init_(3*i + 9) += 0.35*normDist_(gen_);
      }
    }
    else{
      gc_init_.tail(12) << 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;
    }
    if (playerNum_ == 0) {
      gc_init_.head(3) << 1.5 * std::cos(theta), 1.5 * std::sin(theta), 0.5;
      gc_init_.segment(3, 4) << cos((theta - M_PI) / 2), 0, 0, sin((theta - M_PI) / 2);
    }
    else {
      gc_init_.head(3) << 1.5 * std::cos(theta + M_PI), 1.5 * std::sin(theta + M_PI), 0.5;
      gc_init_.segment(3, 4) << cos(theta / 2), 0, 0, sin(theta / 2);
    }
    anymal_->setState(gc_init_, gv_init_);
    anymal_->getState(gc_, gv_);
    if(objIdx_ < 0){
      anymalO_->getState(gcOppo_, gvOppo_);
      gcOppoBias_.setZero();
    }
    else{
      object_ = objPtr;
      raisim::Vec<3> oppoPos, oppoVel;
      object_->getPosition(0, oppoPos);
      object_->getVelocity(0, oppoVel);
      gcOppo_.head(3) = oppoPos.e();
      gvOppo_.head(3) = oppoVel.e();
      gcOppoBias_ << 0.1*normDist_(gen_), 0.1*normDist_(gen_);
    }
    raisim::Vec<4> quat;
    raisim::Mat<3, 3> rot;
    quat[0] = gc_init_[3];
    quat[1] = gc_init_[4];
    quat[2] = gc_init_[5];
    quat[3] = gc_init_[6];
    raisim::quatToRotMat(quat, rot);
    rot_ = rot.e();
    contactInfo_.setZero();

    // reset states
    anymal_->getState(gc_, gv_);
    xyBefore_ << gc_.head(2), gcOppo_.head(2);
    RR_ << gc_.head(2).norm(), gcOppo_.head(2).head(2).norm(), 0, 0;
    discountFactorForSatefy_ = 1.0;

    const Eigen::Vector2d perturbedGcOppo = gcOppo_.head(2);// + gcOppoBias_ +  0.1*Eigen::Vector2d(normDist_(gen_), normDist_(gen_));
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
    Eigen::VectorXf oppoState(6); 
    oppoState << gcOppo_.head(3).cast<float>(), gvOppo_.head(3).cast<float>();
    for(int i = 0; i < 10; i++){
      myHistory_.push_front(state);
      opponentHistory_.push_front(oppoState);
    }
    actionHistory_.clear();
    Eigen::VectorXf action(12); action.setZero();
    for(int i = 0; i < 3; i++) actionHistory_.push_front(action);

    return true;
  }

  bool resetRandom(raisim::World *world) {
    timer_ = 0;
    objIdx_ = -1;
    
    // set random initial position
    const double theta = 2*M_PI*uniDist_(gen_);
    const double r = 2.5*uniDist_(gen_);
    gc_init_.head(3) << r*std::cos(theta), r*std::sin(theta), 0.6 + 0.2*uniDist_(gen_);
    // set perturbed initial orientation
    const double yaw_init = 2*M_PI*uniDist_(gen_);


    Eigen::Matrix3d yawMat, desRot;
    yawMat << cos(yaw_init), -sin(yaw_init), 0,
              sin(yaw_init),  cos(yaw_init), 0,
                      0,         0, 1;
    raisim::Mat<3,3> initialMat, rotMat;
    raisim::Vec<3> axis = {normDist_(gen_), normDist_(gen_), normDist_(gen_)};
    axis /= axis.norm();
    raisim::angleAxisToRotMat(axis, normDist_(gen_) * 0.2, initialMat);
    desRot = yawMat * initialMat.e();

    for(int i = 0; i < 3; i++) for(int j = 0; j < 3; j++) rotMat(i,j) = desRot(i,j);
    raisim::Vec<4> raisimQuat;
    raisim::rotMatToQuat(rotMat, raisimQuat);
    gc_init_.segment(3, 4) = raisimQuat.e(); 

    anymal_->setState(gc_init_, gv_init_);
    anymal_->getState(gc_, gv_);
    if(objIdx_ < 0){
      anymalO_->getState(gcOppo_, gvOppo_);
      gcOppoBias_.setZero();
    }
    
    raisim::Vec<4> quat;
    raisim::Mat<3, 3> rot;
    quat[0] = gc_init_[3];
    quat[1] = gc_init_[4];
    quat[2] = gc_init_[5];
    quat[3] = gc_init_[6];
    raisim::quatToRotMat(quat, rot);
    rot_ = rot.e();
    contactInfo_.setZero();

    // reset states
    anymal_->getState(gc_, gv_);
    xyBefore_ << gc_.head(2), gcOppo_.head(2);
    RR_ << gc_.head(2).norm(), gcOppo_.head(2).head(2).norm(), 0, 0;
    discountFactorForSatefy_ = 1.0;

    const Eigen::Vector2d perturbedGcOppo = gcOppo_.head(2); // + gcOppoBias_ +  0.1*Eigen::Vector2d(normDist_(gen_), normDist_(gen_));
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
    Eigen::VectorXf oppoState(6); 
    oppoState << gcOppo_.head(3).cast<float>(), gvOppo_.head(3).cast<float>();
    for(int i = 0; i < 10; i++){
      myHistory_.push_front(state);
      opponentHistory_.push_front(oppoState);
    }
    actionHistory_.clear();
    Eigen::VectorXf action(12); action.setZero();
    for(int i = 0; i < 3; i++){
      actionHistory_.push_front(action);
    }
    return true;
  }

  void getContactInfo(raisim::World *world){
    contactInfo_.setZero();
    for(auto& contact: anymal_->getContacts()){
      if (contact.skip()) continue;
      for(int i = 0; i < 13; i++){
        if(contact.getPairObjectIndex() != world->getObject("ground")->getIndexInWorld() && contactIndices_[i] == contact.getlocalBodyIndex()){
          if(i==0){
            contactInfo_.head(3) = rot_.transpose()*(contact.getPosition().e() - gc_.head(3));
          }
          else{
            raisim::Vec<3> framePos;
            anymal_->getFramePosition(contactFrames_[i-1], framePos);
            contactInfo_(i + 2) = (contact.getPosition().e() - framePos.e()).norm();
          }
          // std::cout << "contact force = " << (contact.getContactFrame().e().transpose() * contact.getImpulse().e()).transpose() << std::endl;
          // std::cout << "contactInfo_(" << i + 2 << ") = " << contactInfo_(i + 2) << std::endl;  
          break;
        }
      }
    }
  }

  void updateOnlyState(){
    anymal_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    raisim::Mat<3, 3> rot;
    quat[0] = gc_[3];
    quat[1] = gc_[4];
    quat[2] = gc_[5];
    quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    rot_ = rot.e();
    anymalO_->getState(gcOppo_, gvOppo_);
  }

  void updateOnlyState2(){
    anymal_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    raisim::Mat<3, 3> rot;
    quat[0] = gc_[3];
    quat[1] = gc_[4];
    quat[2] = gc_[5];
    quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    rot_ = rot.e();
    if(objIdx_ < 0){
      anymalO_->getState(gcOppo_, gvOppo_);
    }
    else{
      raisim::Vec<3> oppoPos, oppoVel;
      object_->getPosition(0, oppoPos);
      object_->getVelocity(0, oppoVel);
      gcOppo_.head(3) = oppoPos.e();
      gvOppo_.head(3) = oppoVel.e();
    }
  }

  void getOppoState(Eigen::VectorXd& gcOppo, Eigen::VectorXd& gvOppo){
    gcOppo.head(3) = gcOppo_.head(3);
    gvOppo.head(3) = gvOppo_.head(3);
  }

  inline void updateObservation(raisim::World *world) {
    updateOnlyState();
    getContactInfo(world_);
  }

  void fightReward(Reward *rewards){
    bool opponentContact = false;
    int contactCount = 0;
    for(auto& contact: anymal_->getContacts()) {
      if(objIdx_ < 0){
        if(contact.getPairObjectIndex() == anymalO_->getIndexInWorld()) {
          opponentContact = true;
        }
      }
      else{
        if(contact.getPairObjectIndex() == object_->getIndexInWorld()) {
          opponentContact = true;
        }
      }
      contactCount++;
    }
    double fightReward = 0;
    double rpyReward = 0;
    double closeReward = 0;

    double rDiff = std::min(RR_(0) - RR_(1) + 0.1, 1.5);
    if(rDiff > 0.0){// && RR_(0) > 2.2){
      rewards->record("j_rDiff", 0.05*rDiff*rDiff);
    }
    else rewards->record("j_rDiff", 0.0);

    // kinematic energy
    if(gc_(2) + 0.18 < gcOppo_(2) && (gc_.head(2) - gcOppo_.head(2)).norm() < 0.3){
      // if I am under the opponent and close to the opponent,
      // maximize kinematic energy
      double E = 0.05*gv_(2)*gv_(2) + (gc_(2) - 0.12);
      closeReward += 0.5*E*E;
    }
    else{
      closeReward -= 0.25*(gc_(2) - 0.12)*(gc_(2) - 0.12);
    }

    if(opponentContact){ // contact with opponent or close to the opponent
      // 1) just contact
      fightReward += 0.02; // contact reward(generally good)

      // 2) I should not on the opponent
      const double heightDiff = gc_(2) - gcOppo_(2) - 0.12;
      if(heightDiff > 0){
        fightReward += heightDiff*heightDiff;
      }
      // // 2) move away from the center
      // fightReward += 2.5*(gcOppo_.head(2).norm() - xyBefore_.segment(2,2).norm());
      // // 3) my move along to the (past gc to past opponent) line
      // const Eigen::Vector2d me2oppo_before = (xyBefore_.segment(2,2) - xyBefore_.head(2)).normalized();
      // const Eigen::Vector2d oppoDelta = gcOppo_.head(2) - xyBefore_.segment(2,2);
      // const double myDelta = oppoDelta.dot(me2oppo_before); 
      // if(myDelta < 0) fightReward += 2*myDelta;
      // else            fightReward +=   myDelta;
      
      // consider opponent's rpy, vz, my rpy
      if(objIdx_ < 0){
        if(gcOppo_.segment(3,4).isZero()){std::cout << "gcOppo_ quat is zerro" << std::endl;}
        else{
          Eigen::Vector2d rpyOppo = quatToRPY(gcOppo_.segment(3,4)).head(2);
          rpyOppo.cwiseAbs();
          rpyOppo(0) = std::min(std::max(rpyOppo(0) - 0.2, 0.0), 0.8);
          rpyOppo(1) = std::min(std::max(rpyOppo(1) - 0.2, 0.0), 0.8);
          Eigen::Vector2d meToOppo = (gcOppo_.head(2) - gc_.head(2)).normalized();
          rpyReward += 0.8*rpyOppo.squaredNorm() + gvOppo_(2)*gvOppo_(2);
        }
      }
    }
    Eigen::Vector2d rpyMy = quatToRPY(gc_.segment(3,4)).head(2);
    rpyMy.cwiseAbs();
    rpyMy(0) = std::min(std::max(rpyMy(0) - 0.3, 0.0), 1.2);
    rpyMy(1) = std::min(std::max(rpyMy(1) - 0.3, 0.0), 1.2);
    rpyReward -= 3*rpyMy.squaredNorm();

    fightReward = fightReward*(1 + (fightReward > 0)*(discountFactorForSatefy_-1));
    rewards->record("d_fight", fightReward);
    rewards->record("i_rpy", rpyReward);
    rewards->record("k_closeStrategy", closeReward);
  }

  void desiredReward(Reward *rewards){
    float actionSmooth = (actionHistory_[0] - actionHistory_[1]).squaredNorm()*(timer_ > 1);
    actionSmooth += (actionHistory_[0] - 2 * actionHistory_[1] + actionHistory_[2]).squaredNorm()*(timer_ > 2);
    if(isnan(actionSmooth)){
      for(int i = 0; i < 3; i++){
        std::cout << "action" << i << " = " << actionHistory_[i].transpose() << std::endl;
      }
    }
    rewards->record("f_smooth", double(actionSmooth));
  }

  inline void recordReward(Reward *rewards) {
    updateOnlyState();
    double velNorm = std::pow(std::max(gv_.head(6).norm() - 0.5, 0.0), 2)*(3 + gc_.head(3).squaredNorm());
    rewards->record("a_forwardVel", velNorm);
    rewards->record("b_torque", anymal_->getGeneralizedForce().squaredNorm());
    RR_ << gc_.head(2).norm(), gcOppo_.head(2).norm(), 0, 0;
    double safety = std::pow(std::max(RR_(0) - 1.0, 0.0), 2);
    safety += 20*std::pow(std::max(RR_(0) - 2.0, 0.0), 4);
    discountFactorForSatefy_ = 1 - (RR_(0) > 2.5)*(RR_(0) - 2.5)/0.5; 
    rewards->record("c_safety", safety);
    fightReward(rewards);
    desiredReward(rewards);

    // how much move along to the (past gc to current opponent) line
    Eigen::Vector2d pastToOppo = (gcOppo_.head(2) - myHistory_[5].head(2)).normalized();
    Eigen::Vector2d pastToCur  = gc_.head(2) - myHistory_[5].head(2);
    double relative = pastToCur.dot(pastToOppo);
    rewards->record("e_relative", relative*(1 + (relative > 0)*(discountFactorForSatefy_ - 1)));
    rewards->record("g_survive", 0);
    double r = (gc_.head(2) - gcOppo_.head(2)).norm() - minObjSize_;
    if(r < 0.0) rewards->record("h_distance", 50*r*r);
    else        rewards->record("h_distance",   r*r);
    xyBefore_ << gc_.head(2), gcOppo_.head(2);
  }

  inline const Eigen::VectorXd &getObservation() {
    const Eigen::Vector2d perturbedGcOppo = gcOppo_.head(2); // + gcOppoBias_ + (0.2 + 0.8*(objIdx_ > 0))*0.1*Eigen::Vector2d(normDist_(gen_), normDist_(gen_));
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
    // state(2) += 0.1*normDist_(gen_);

    obDouble_ << state, myHistory_[0], myHistory_[1], actionHistory_[0].cast<double>();
    // for(int i = 0; i < obDim_; i++){
    //   if(isnan(obDouble_(i))){
    //     std::cout << "obDouble_(" << i << ") is nan" << std::endl;
    //     std::cout << "timer_ = " << timer_ << std::endl;
    //     std::cout << "objidx = " << objIdx_ << std::endl;
    //   }
    // }
    // queue, pop
    myHistory_.push_front(state);
    myHistory_.pop_back();


    // opponentHistory_.push_front(gcOppo_.head(3).cast<float>());
    // opponentHistory_.pop_back();
    return obDouble_;
  }

  void setName(const std::string &name) {
    name_ = name;
  }

  void setOpponentName(const std::string &name) {
    opponentName_ = name;
  }

  void setPlayerNum(const int &playerNum) {
    playerNum_ = playerNum;
  }

  inline bool isTerminalState(raisim::World *world) {
    for (auto &contact: anymal_->getContacts()) {
      if (footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end()) {
        return true;
      }
    }
    return false;
  }

  inline int getObDim() {
    return obDim_;
  }

  inline int getActionDim() {
    return actionDim_;
  }


 private:
  std::string name_, opponentName_;
  int gcDim_, gvDim_, nJoints_, playerNum_ = 0;
  raisim::ArticulatedSystem *anymal_, *anymalO_;
  raisim::World *world_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_;
  int obDim_ = 0, actionDim_ = 0;
  double forwardVelRewardCoeff_ = 0.;
  double torqueRewardCoeff_ = 0.;
  //
  bool pdStandingMode_ = false;
  // additional initial settings
  std::vector<size_t> contactIndices_, contactFrames_;
  // opponent related
  int objIdx_;
  raisim::SingleBodyObject* object_;
  Eigen::VectorXd gcOppo_, gvOppo_;
  Eigen::Vector2d gcOppoBias_;
  // maintained variables for reward calculation
  int timer_;
  Eigen::Matrix3d rot_;
  Eigen::Vector4d xyBefore_, RR_;
  Eigen::VectorXd contactInfo_;
  double discountFactorForSatefy_;
  const double minObjSize_ = 0.3;
  // history related
  std::deque<Eigen::VectorXf> opponentHistory_;
  std::deque<Eigen::VectorXf> actionHistory_;
  std::deque<Eigen::VectorXd> myHistory_;

  
  thread_local static std::normal_distribution<double> normDist_;
  thread_local static std::uniform_real_distribution<double> uniDist_;
  thread_local static std::mt19937 gen_;
};
thread_local std::mt19937 raisim::AnymalController_20190430::gen_;
thread_local std::normal_distribution<double> raisim::AnymalController_20190430::normDist_(0., 1.);
thread_local std::uniform_real_distribution<double> raisim::AnymalController_20190430::uniDist_(0., 1.);
}