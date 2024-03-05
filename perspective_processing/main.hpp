double homography_data_juan_daniel_hd[9] = {
        -2.0291847267655947, -7.437357958160159, 906.3678688014978,
        -0.23139812416697048,-12.90401791851415, 1352.4388037555063,
        -0.0005427455069292742, -0.022938638867020866,1.0
};

double homography_data_juan_daniel_hd_not_parking[9] = {
        -2.9570130516412307, -7.486029329997402, 1199.5761134071247,
        -0.21683788651694957, -13.85110669920207, 1788.6994055593027,
        -0.0005427401944538209, -0.022939173454083684, 1.0
};


const int IMG_WIDTH = 640;
const int IMG_HEIGHT = 480;
const int X_DIST_CALIB = 680;  // distance to the end of the ROI in pixels from the rear wheel axis
const int X_FAR = 680;      // pixels in car coordinates (in the future in mm)
const int X_CLOSE = 320;    // pixels in car coordinates (in the future in mm)