PinholeCamera::PinholeCamera(const PinholeCamera::Parameters& params)
 : mParameters(params)
{
    if ((mParameters.k1() == 0.0) &&
        (mParameters.k2() == 0.0) &&
        (mParameters.p1() == 0.0) &&
        (mParameters.p2() == 0.0))
    {
        m_noDistortion = true;
    }
    else
    {
        m_noDistortion = false;
    }

    // Inverse camera projection matrix parameters
    m_inv_K11 = 1.0 / mParameters.fx();
    m_inv_K13 = -mParameters.cx() / mParameters.fx();
    m_inv_K22 = 1.0 / mParameters.fy();
    m_inv_K23 = -mParameters.cy() / mParameters.fy();
}



/**
 * \brief Lifts a point from the image plane to its projective ray
 *
 * \param p image coordinates
 * \param P coordinates of the projective ray
 */
void
PinholeCamera::liftProjective(const Eigen::Vector2d& p, Eigen::Vector3d& P) const
{
    double mx_d, my_d,mx2_d, mxy_d, my2_d, mx_u, my_u;
    double rho2_d, rho4_d, radDist_d, Dx_d, Dy_d, inv_denom_d;
    //double lambda;

    // Lift points to normalised plane
    mx_d = m_inv_K11 * p(0) + m_inv_K13;
    my_d = m_inv_K22 * p(1) + m_inv_K23;

    if (m_noDistortion)
    {
        mx_u = mx_d;
        my_u = my_d;
    }
    else
    {
        if (0)
        {
            double k1 = mParameters.k1();
            double k2 = mParameters.k2();
            double p1 = mParameters.p1();
            double p2 = mParameters.p2();

            // Apply inverse distortion model
            // proposed by Heikkila
            mx2_d = mx_d*mx_d;
            my2_d = my_d*my_d;
            mxy_d = mx_d*my_d;
            rho2_d = mx2_d+my2_d;
            rho4_d = rho2_d*rho2_d;
            radDist_d = k1*rho2_d+k2*rho4_d;
            Dx_d = mx_d*radDist_d + p2*(rho2_d+2*mx2_d) + 2*p1*mxy_d;
            Dy_d = my_d*radDist_d + p1*(rho2_d+2*my2_d) + 2*p2*mxy_d;
            inv_denom_d = 1/(1+4*k1*rho2_d+6*k2*rho4_d+8*p1*my_d+8*p2*mx_d);

            mx_u = mx_d - inv_denom_d*Dx_d;
            my_u = my_d - inv_denom_d*Dy_d;
        }
        else
        {
            // Recursive distortion model
            int n = 8;
            Eigen::Vector2d d_u;
            distortion(Eigen::Vector2d(mx_d, my_d), d_u);
            // Approximate value
            mx_u = mx_d - d_u(0);
            my_u = my_d - d_u(1);

            for (int i = 1; i < n; ++i)
            {
                distortion(Eigen::Vector2d(mx_u, my_u), d_u);
                mx_u = mx_d - d_u(0);
                my_u = my_d - d_u(1);
            }
        }
    }

    // Obtain a projective ray
    P << mx_u, my_u, 1.0;
}


/**
 * \brief Apply distortion to input point (from the normalised plane)
 *
 * \param p_u undistorted coordinates of point on the normalised plane
 * \return to obtain the distorted point: p_d = p_u + d_u
 */
void
PinholeCamera::distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u) const
{
    double k1 = mParameters.k1();
    double k2 = mParameters.k2();
    double p1 = mParameters.p1();
    double p2 = mParameters.p2();

    double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;

    mx2_u = p_u(0) * p_u(0);
    my2_u = p_u(1) * p_u(1);
    mxy_u = p_u(0) * p_u(1);
    rho2_u = mx2_u + my2_u;
    rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
    d_u << p_u(0) * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u),
           p_u(1) * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);
}



void delByDis(const vector<cv::Point2f> &un_cur_pts, const vector<cv::Point2f> &un_forw_pts, int cur_seq, vector<uchar> &status) {
    std::ifstream file("/media/nyamori/8856D74A56D73820/vslam/dataset/kitti/data_odometry_poses/dataset/poses/05.txt");  // Replace with your file name

    assert(file);

    Eigen::Matrix<double, 3, 3> Ri, Rj;
    Eigen::Matrix<double, 3, 1> ti, tj;

    std::string line1, line2;
    for(int i = 0; i <= cur_seq; i++) std::getline(file, line1);
    
    std::istringstream iss1(line1);
    for(int i = 0; i < 3; i++ ) {
        for(int j = 0; j < 4; j++) {
            if(j < 3) iss1 >> Ri(i, j);
            else iss1 >> ti(i);
        }
    }
    std::getline(file, line2);
    std::istringstream iss2(line2);
    for(int i = 0; i < 3; i++ ) {
        for(int j = 0; j < 4; j++) {
            if(j < 3) iss2 >> Rj(i, j);
            else iss2 >> tj(i, 0);
        }
    }

    Ri = Rj.transpose() * Ri;
    ti = Rj.transpose() * (ti - tj);

    double A = Ri(0, 2), B = Ri(1, 2), x0, y0, dis;
    
    for(int i = 0; i < un_cur_pts.size(); i++) {
        x0 = Ri(0, 0) * un_cur_pts[i].x + Ri(0, 1) * un_cur_pts[i].y + ti(0, 0);
        y0 = Ri(1, 0) * un_cur_pts[i].x + Ri(1, 1) * un_cur_pts[i].y + ti(1, 0);
        dis = fabs((un_forw_pts[i].x-x0)*B - (un_forw_pts[i].y-y0)*A) / std::sqrt(A*A + B*B);

        double xx = un_cur_pts[i].x - un_forw_pts[i].x;
        double yy = un_cur_pts[i].y - un_forw_pts[i].y;
        if(xx*xx+yy*yy < 0.5) status.push_back(0); 
        else {
            if(dis > 5.0) status.push_back(0); else status.push_back(1);
        }
    }
}