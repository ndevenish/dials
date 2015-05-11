/*
 * rgb_2d.h
 *
 *  Copyright (C) 2015 Diamond Light Source
 *
 *  Author: Luis Fuentes-Montero (Luiso)
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */

#ifndef DIALS_RGB_IMG_BUILDER_H
#define DIALS_RGB_IMG_BUILDER_H
#include <iostream>
#include <string>
#include <scitbx/array_family/flex_types.h>
#include <cmath>

namespace dials { namespace viewer { namespace boost_python {
  using scitbx::af::flex_double;
  using scitbx::af::flex_int;
  using scitbx::af::flex_grid;

  int get_digits( double nm, int (&dgt_num)[12]){

    int err_cod = 0;

    char asc_str[12];

    std::string str;
    std::cout << "nm =" << nm << "\n";
    sprintf (asc_str, "           ");

    sprintf (asc_str, "%g", nm);
    std::cout << "asc_str = <<" << asc_str << ">>\n\n";
    str = asc_str;

    for (int i = 0; i < 12; i++){
      str = asc_str[i];
      dgt_num[i] = asc_str[i] - 48;

      if( dgt_num[i] < 0 or dgt_num[i] > 9 ){
        if( str == "." ){
          dgt_num[i] = 10;
        }else if( str == "e" ){
          dgt_num[i] = 11;
        }else if( str == "+" ){
          dgt_num[i] = 12;
        }else if( str == "-" ){
          dgt_num[i] = 13;
        }else if( str == " " ){
          dgt_num[i] = 14;
        }else if( asc_str[i] == 0 ){
          dgt_num[i] = 15;
        }else{
          std::cout << "\nfound \'" << str << "\' and not converted";
          err_cod = asc_str[i];
          std::cout << "\n char(" << err_cod << ") found\n";
          err_cod = 1;
        }
      }

    std::cout << " {" << asc_str[i] << "} " << "["<< dgt_num[i] <<"],";
    }

    std::cout << "\nerr_cod =" << err_cod << "\n";

    return err_cod;
  }


  flex_int gen_str_tst(flex_double & data_num) {

    flex_int bmp_dat(flex_grid<>(7, 7, 10),0);
    int npos=data_num.accessor().all()[0];
    int digit_val[12];
    int err_conv = 0;

    double dl_nm;



    for (int pos = 0; pos < npos; pos++) {
      dl_nm = data_num(pos, 0);

      err_conv = get_digits(dl_nm, digit_val);
      if(err_conv == 1){
        std::cout << "\nerror converting\n";
      }

      std::cout << "\n______________________________________\n";


    }

    return bmp_dat;
  }




  flex_int gen_img(flex_double & data2d) {

    int ndept=data2d.accessor().all()[0];
    flex_int bmp_dat(flex_grid<>(7, 7, ndept),0);

    int arr_2d_0[7][7] = {{0,0,0,0,0,0,0},
                          {0,0,1,1,1,1,0},
                          {0,1,1,0,0,1,1},
                          {0,1,0,0,1,0,1},
                          {0,1,0,1,0,0,1},
                          {0,0,1,1,1,1,0},
                          {0,0,0,0,0,0,0}};
    for (int row = 0; row < 7 ; row++) {
      for (int col = 0; col < 7; col++) {
        bmp_dat(col, row, 0) = arr_2d_0[col][row];
      }
    }

    int arr_2d_1[7][7] = {{0,0,0,0,0,0,0},
                          {0,1,1,1,0,0,0},
                          {0,0,0,1,0,0,0},
                          {0,0,0,1,0,0,0},
                          {0,0,0,1,0,0,0},
                          {0,1,1,1,1,1,0},
                          {0,0,0,0,0,0,0}};
    for (int row = 0; row < 7 ; row++) {
      for (int col = 0; col < 7; col++) {
        bmp_dat(col, row, 1) = arr_2d_1[col][row];
      }
    }

    int arr_2d_2[7][7] = {{0,0,0,0,0,0,0},
                          {0,0,1,1,1,1,0},
                          {0,1,0,0,0,1,1},
                          {0,0,0,0,1,1,0},
                          {0,0,0,1,1,0,0},
                          {0,1,1,1,1,1,1},
                          {0,0,0,0,0,0,0}};
    for (int row = 0; row < 7 ; row++) {
      for (int col = 0; col < 7; col++) {
        bmp_dat(col, row, 2) = arr_2d_2[col][row];
      }
    }

    int arr_2d_3[7][7] = {{0,0,0,0,0,0,0},
                          {0,0,1,1,1,1,0},
                          {0,1,0,0,0,1,1},
                          {0,0,0,1,1,1,0},
                          {0,0,0,0,0,1,1},
                          {0,1,1,1,1,1,0},
                          {0,0,0,0,0,0,0}};
    for (int row = 0; row < 7 ; row++) {
      for (int col = 0; col < 7; col++) {
        bmp_dat(col, row, 3) = arr_2d_3[col][row];
      }
    }

    int arr_2d_4[7][7] = {{0,0,0,0,0,0,0},
                          {0,0,0,0,1,1,0},
                          {0,0,0,1,1,1,0},
                          {0,0,1,1,0,1,0},
                          {0,1,1,1,1,1,1},
                          {0,0,0,0,0,1,0},
                          {0,0,0,0,0,0,0}};
    for (int row = 0; row < 7 ; row++) {
      for (int col = 0; col < 7; col++) {
        bmp_dat(col, row, 4) = arr_2d_4[col][row];
      }
    }

    int arr_2d_5[7][7] = {{0,0,0,0,0,0,0},
                          {0,1,1,1,1,1,0},
                          {0,1,0,0,0,0,0},
                          {0,1,1,1,1,1,1},
                          {0,0,0,0,0,0,1},
                          {0,1,1,1,1,1,1},
                          {0,0,0,0,0,0,0}};
    for (int row = 0; row < 7 ; row++) {
      for (int col = 0; col < 7; col++) {
        bmp_dat(col, row, 5) = arr_2d_5[col][row];
      }
    }

    int arr_2d_6[7][7] = {{0,0,0,0,0,0,0},
                          {0,0,1,1,1,1,0},
                          {0,1,1,0,0,0,0},
                          {0,1,1,1,1,1,1},
                          {0,1,1,0,0,0,1},
                          {0,0,1,1,1,1,1},
                          {0,0,0,0,0,0,0}};
    for (int row = 0; row < 7 ; row++) {
      for (int col = 0; col < 7; col++) {
        bmp_dat(col, row, 6) = arr_2d_6[col][row];
      }
    }

    int arr_2d_7[7][7] = {{0,0,0,0,0,0,0},
                          {0,1,1,1,1,1,1},
                          {0,0,0,0,0,1,1},
                          {0,0,0,0,1,1,0},
                          {0,0,0,1,1,0,0},
                          {0,0,1,1,0,0,0},
                          {0,0,0,0,0,0,0}};
    for (int row = 0; row < 7 ; row++) {
      for (int col = 0; col < 7; col++) {
        bmp_dat(col, row, 7) = arr_2d_7[col][row];
      }
    }

    int arr_2d_8[7][7] = {{0,0,0,0,0,0,0},
                          {0,0,1,1,1,1,0},
                          {0,1,1,0,0,1,1},
                          {0,0,1,1,1,1,0},
                          {0,1,1,0,0,1,1},
                          {0,0,1,1,1,1,0},
                          {0,0,0,0,0,0,0}};
    for (int row = 0; row < 7 ; row++) {
      for (int col = 0; col < 7; col++) {
        bmp_dat(col, row, 8) = arr_2d_8[col][row];
      }
    }

    int arr_2d_9[7][7] = {{0,0,0,0,0,0,0},
                          {0,0,1,1,1,1,0},
                          {0,1,1,0,0,1,1},
                          {0,1,1,1,1,1,1},
                          {0,0,0,0,0,1,1},
                          {0,1,1,1,1,1,0},
                          {0,0,0,0,0,0,0}};
    for (int row = 0; row < 7 ; row++) {
      for (int col = 0; col < 7; col++) {
        bmp_dat(col, row, 9) = arr_2d_9[col][row];
      }
    }

    int arr_2d_point[7][7] = {{0,0,0,0,0,0,0},
                              {0,0,0,0,0,0,0},
                              {0,0,0,0,0,0,0},
                              {0,0,0,0,0,0,0},
                              {0,0,1,1,0,0,0},
                              {0,0,1,1,0,0,0},
                              {0,0,0,0,0,0,0}};
    for (int row = 0; row < 7 ; row++) {
      for (int col = 0; col < 7; col++) {
        bmp_dat(col, row, 10) = arr_2d_point[col][row];
      }
    }

    int arr_2d_e[7][7] = {{0,0,0,0,0,0,0},
                          {0,0,1,1,1,1,0},
                          {0,1,1,0,0,1,0},
                          {0,1,1,1,1,1,0},
                          {0,1,1,0,0,0,0},
                          {0,0,1,1,1,1,0},
                          {0,0,0,0,0,0,0}};
    for (int row = 0; row < 7 ; row++) {
      for (int col = 0; col < 7; col++) {
        bmp_dat(col, row, 11) = arr_2d_e[col][row];
      }
    }

    int arr_2d_plus[7][7] =  {{0,0,0,0,0,0,0},
                              {0,0,0,1,0,0,0},
                              {0,0,0,1,0,0,0},
                              {0,1,1,1,1,1,0},
                              {0,0,0,1,0,0,0},
                              {0,0,0,1,0,0,0},
                              {0,0,0,0,0,0,0}};
    for (int row = 0; row < 7 ; row++) {
      for (int col = 0; col < 7; col++) {
        bmp_dat(col, row, 12) = arr_2d_plus[col][row];
      }
    }

    int arr_2d_minus[7][7] = {{0,0,0,0,0,0,0},
                              {0,0,0,0,0,0,0},
                              {0,0,0,0,0,0,0},
                              {0,1,1,1,1,1,0},
                              {0,0,0,0,0,0,0},
                              {0,0,0,0,0,0,0},
                              {0,0,0,0,0,0,0}};
    for (int row = 0; row < 7 ; row++) {
      for (int col = 0; col < 7; col++) {
        bmp_dat(col, row, 13) = arr_2d_minus[col][row];
      }
    }

    int arr_2d_space[7][7] = {{0,0,0,0,0,0,0},
                              {0,0,0,0,0,0,0},
                              {0,0,0,0,0,0,0},
                              {0,0,0,0,0,0,0},
                              {0,0,0,0,0,0,0},
                              {0,0,0,0,0,0,0},
                              {0,0,0,0,0,0,0}};
    for (int row = 0; row < 7 ; row++) {
      for (int col = 0; col < 7; col++) {
        bmp_dat(col, row, 14) = arr_2d_space[col][row];
      }
    }

    int arr_2d_null[7][7] =  {{0,0,0,0,0,0,0},
                              {0,0,0,0,0,0,0},
                              {0,0,0,0,0,0,0},
                              {0,0,0,0,0,0,0},
                              {0,0,0,0,0,0,0},
                              {0,0,0,0,0,0,0},
                              {0,0,0,0,0,0,0}};
    for (int row = 0; row < 7 ; row++) {
      for (int col = 0; col < 7; col++) {
        bmp_dat(col, row, 15) = arr_2d_null[col][row];
      }
    }

    return bmp_dat;
  }


  class rgb_img
  {

    private:
      int red_byte[255 * 3 + 1];
      int green_byte[255 * 3 + 1];
      int blue_byte[255 * 3 + 1];

    public:
      rgb_img() {

        for (int i = 0; i < 255; i++){
          red_byte[i] = i;
          green_byte[i + 255] = i;
          blue_byte[i + 255 * 2 ] = i;
        }

        for (int i = 255; i < 255 * 3; i++){
          red_byte[i] = 255;
        }

        for (int i = 0; i < 255; i++){
          green_byte[i] = 0;
        }

        for (int i = 255 * 2; i < 255 * 3; i++){
          green_byte[i] = 255;
        }

        for (int i = 0; i < 255 * 2; i++){
          blue_byte[i] = 0;
        }

        blue_byte[765] = 255;
        green_byte[765] = 255;
        red_byte[765] = 255;


      }

      flex_int gen_bmp(flex_double & data2d) {

        int ncol=data2d.accessor().all()[1];
        int nrow=data2d.accessor().all()[0];

        double max = 1, min = -1, loc_cel, dif = 0;
        std::cout << "\n here 01 \n";
        flex_double scaled_array(flex_grid<>(nrow, ncol),0);
        std::cout << "\n here 02 \n";
        for (int row = 0; row < nrow ; row++) {
          for (int col = 0; col < ncol; col++) {

            loc_cel = data2d(row, col);
            if(row == 0 and col == 0){
              max = loc_cel;
              min = loc_cel;
            }else{
              if(loc_cel > max){
                max = loc_cel;
              }
              if(loc_cel < min){
                min = loc_cel;
              }
            }
            scaled_array(row, col) = data2d(row, col);
          }
        }

        std::cout << "\n max = "<< max << "\n";
        std::cout << "\n min = "<< min << "\n \n";

        dif = max - min;

        for (int row = 0; row < nrow ; row++) {
          for (int col = 0; col < ncol; col++) {
            scaled_array(row, col) = 255.0 * 3 * (( scaled_array(row, col) - min)
                                                          / dif );
          }
        }

        std::cout << "\n here 03 \n";
        bool auto_zoom = false;
        int px_scale = 0;

        if(ncol < 200 and nrow < 200){
          auto_zoom = true;
        }

        float diagn;
        if(auto_zoom == true){
          if(ncol < 20 and nrow < 20){
            px_scale = 50;
          }else{
            diagn = sqrt(ncol * ncol + nrow * nrow);
            px_scale = (1000.0 / diagn);
          }


        }else{
          px_scale = 1;
        }



          flex_int bmp_dat(flex_grid<>(nrow * px_scale, ncol * px_scale, 3),0);
                  std::cout << "\n auto_zoom == true \n" << "px_scale ="
                            << px_scale << "\n";


        for (int row = 0; row < nrow ; row++) {
          for (int col = 0; col < ncol; col++) {
            if(auto_zoom == true){

              for(int pix_row = row * px_scale;
                  pix_row < row * px_scale + px_scale;
                  pix_row++){
                for(int pix_col = col * px_scale;
                    pix_col < col * px_scale + px_scale;
                    pix_col++){

                  bmp_dat(pix_row, pix_col, 0) = red_byte[int(
                                                 scaled_array(row, col))];
                  bmp_dat(pix_row, pix_col, 1) = green_byte[int(
                                                 scaled_array(row, col))];
                  bmp_dat(pix_row, pix_col, 2) = blue_byte[int(
                                                 scaled_array(row, col))];
                }
              }

            }else{
              bmp_dat(row, col, 0) = red_byte[int(scaled_array(row, col))];
              bmp_dat(row, col, 1) = green_byte[int(scaled_array(row, col))];
              bmp_dat(row, col, 2) = blue_byte[int(scaled_array(row, col))];
            }

          }
        }
        std::cout << "\n here 04 \n";

      return bmp_dat;
      }

  };


}}}

#endif
