//
// Created by jerett on 2018-12-24.
//

#include "tracker.h"


namespace cd {

cv::Ptr<Tracker> Tracker::Create(cd::Tracker::Algorithm algorithm) {
    cv::Ptr<Tracker> tracker;
    if (algorithm == Algorithm::Boosting ||
        algorithm == Algorithm::KCF ||
        algorithm == Algorithm::MIL ||
        algorithm == Algorithm::TLD ||
        algorithm == Algorithm::MedianFlow ||
        algorithm == Algorithm::GOTURN || 
        algorithm == Algorithm::OpenCVMOSSE
        ) {
        tracker = cv::Ptr<Tracker>(new TrackerOpenCV(algorithm));
    } else if (algorithm == Algorithm::Staple) {
        tracker = cv::Ptr<Tracker>(new TrackerStaple());
    } else if (algorithm == Algorithm::MOSSE) {
        tracker = cv::Ptr<Tracker>(new TrackerMOSSE());
    }
    assert(tracker != nullptr);
    return tracker;
}

/**
 *  implement opencv tracker.
 */

TrackerOpenCV::TrackerOpenCV(Algorithm type) {
    if (type == Algorithm::Boosting) {
        tracker_ = cv::TrackerBoosting::create();
    } else if (type == Algorithm::KCF) {
        cv::TrackerKCF::Params p;
        p.desc_npca = cv::TrackerKCF::GRAY | cv::TrackerKCF::CN;
        tracker_ = cv::TrackerKCF::create(p);
    } else if (type == Algorithm::MIL) {
        tracker_ = cv::TrackerMIL::create();
    } else if (type == Algorithm::TLD) {
        tracker_ = cv::TrackerTLD::create();
    } else if (type == Algorithm::MedianFlow) {
        tracker_ = cv::TrackerMedianFlow::create();
    } else if (type == Algorithm::GOTURN) {
        tracker_ = cv::TrackerGOTURN::create();
    } else if (type == Algorithm::OpenCVMOSSE) {
        tracker_ = cv::TrackerMOSSE::create();
    } else {
        abort();
    }
    assert(tracker_ != nullptr);
}

bool TrackerOpenCV::Init(cv::Mat &frame, const cv::Rect2d &box) {
    return tracker_->init(frame, box);
}

bool TrackerOpenCV::Update(cv::Mat &frame, cv::Rect2d &out_box) {
    return tracker_->update(frame, out_box);
}


/**
 *  implement staple tracker.
 */

TrackerStaple::TrackerStaple() {
}

bool TrackerStaple::Init(cv::Mat &frame, const cv::Rect2d &box) {
    tracker_.init(frame, box);
    return true;
}

bool TrackerStaple::Update(cv::Mat &frame, cv::Rect2d &out_box) {
    // out_box = tracker_.tracker_staple_update(frame);
    // tracker_.tracker_staple_train(frame, false);
    return tracker_.update(frame, out_box);
}


}