#include <math.h>
#include <stdio.h>
#include <stdlib.h>
//#include <cstring>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
#ifndef MIN
#define MIN(a, b)    ( (a) > (b) ? (b) : (a) )
#endif
#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif
#define CLIP3(x,min,max)         ( (x)< (min) ? (min) : ((x)>(max)?(max):(x)) )


#define int8    char
#define uint8   unsigned char
#define int16   short
#define uint16  unsigned short
#define int32   int
#define uint32  unsigned int
#define int64   long long
#define uint64  unsigned long long

uint8 inv_gamma_table[256] ={
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,
        2,2,3,3,3,3,3,4,4,4,4,5,5,5,5,6,
        6,6,7,7,7,8,8,8,9,9,9,10,10,10,11,11,
        12,12,12,13,13,14,14,15,15,16,16,17,17,18,18,19,
        19,20,20,21,22,22,23,23,24,25,25,26,26,27,28,28,
        29,30,30,31,32,33,33,34,35,36,36,37,38,39,39,40,
        41,42,43,44,44,45,46,47,48,49,50,51,51,52,53,54,
        55,56,57,58,59,60,61,62,63,64,65,66,67,68,70,71,
        72,73,74,75,76,77,78,80,81,82,83,84,86,87,88,89,
        91,92,93,94,96,97,98,100,101,102,104,105,106,108,109,110,
        112,113,115,116,117,119,120,122,123,125,126,128,129,131,132,134,
        135,137,139,140,142,143,145,147,148,150,152,153,155,157,158,160,
        162,163,165,167,169,170,172,174,176,177,179,181,183,185,187,188,
        190,192,194,196,198,200,202,204,206,208,210,212,214,216,218,220,
        222,224,226,228,230,232,234,236,238,240,242,245,247,249,251,253
};
uint8 gamma_045_table[256] = {
        0,21,28,34,39,43,47,50,53,56,59,62,64,66,69,71,
        73,75,77,79,81,83,84,86,88,89,91,93,94,96,97,99,
        100,101,103,104,105,107,108,109,111,112,113,114,115,117,118,119,
        120,121,122,123,124,126,127,128,129,130,131,132,133,134,135,136,
        137,138,139,140,140,141,142,143,144,145,146,147,148,149,149,150,
        151,152,153,154,155,155,156,157,158,159,159,160,161,162,163,163,
        164,165,166,166,167,168,169,169,170,171,172,172,173,174,175,175,
        176,177,177,178,179,179,180,181,182,182,183,184,184,185,186,186,
        187,188,188,189,190,190,191,191,192,193,193,194,195,195,196,196,
        197,198,198,199,200,200,201,201,202,203,203,204,204,205,206,206,
        207,207,208,208,209,210,210,211,211,212,212,213,214,214,215,215,
        216,216,217,217,218,219,219,220,220,221,221,222,222,223,223,224,
        224,225,225,226,227,227,228,228,229,229,230,230,231,231,232,232,
        233,233,234,234,235,235,236,236,237,237,238,238,239,239,240,240,
        241,241,242,242,242,243,243,244,244,245,245,246,246,247,247,248,
        248,249,249,250,250,250,251,251,252,252,253,253,254,254,255,255
};
void apply_invert_gamma(uint8* input,uint8* output,int img_width,int img_height,int img_channels){

    for(uint16 y = 0;y < img_height;y++){
        for(uint16 x =0;x < img_width;x++){
            uint8 temp_r = input[img_channels*(y*img_width+x)];
            uint8 temp_g = input[img_channels*(y*img_width+x)+1];
            uint8 temp_b = input[img_channels*(y*img_width+x)+2];
            output[img_channels*(y*img_width+x)] = inv_gamma_table[temp_r];
            output[img_channels*(y*img_width+x)+1] = inv_gamma_table[temp_g];
            output[img_channels*(y*img_width+x)+2] = inv_gamma_table[temp_b];
        }
    }
    return;
}
void apply_gamma(uint8* input,int img_width,int img_height,int img_channels){

    for(uint16 y = 0;y < img_height;y++){
        for(uint16 x =0;x < img_width;x++){
            uint8 temp_r = input[img_channels*(y*img_width+x)];
            uint8 temp_g = input[img_channels*(y*img_width+x)+1];
            uint8 temp_b = input[img_channels*(y*img_width+x)+2];
            input[img_channels*(y*img_width+x)] = gamma_045_table[temp_r];
            input[img_channels*(y*img_width+x)+1] = gamma_045_table[temp_g];
            input[img_channels*(y*img_width+x)+2] = gamma_045_table[temp_b];
        }
    }
    return;
}

#define  HISTOGRAM_SIZE 256
typedef struct
{
    uint8   enable_auto_contrast;
    uint8   black_percentage;
    uint8   white_percentage;
    uint8   auto_black_min;
    uint8   auto_black_max;
    uint8   auto_white_prc_target;
    uint8   low_contrast;
    uint8   high_contrast;
}auto_contrast_para;

uint8 hist_stats(uint8* input,int img_width,int img_height,uint32 *hist){

    //stats G
    for(int i = 0;i < HISTOGRAM_SIZE;i++){
        for(int y = 0;y < img_height;y++){
            for(int x =0;x < img_width;x++){
                if(i==input[y*img_width+x])
                    hist[i]++;
            }
        }

    }
    return 0;
}

uint8 auto_contrast(uint8* input,uint32* cumu_hist,auto_contrast_para auto_contrast_default_para ,int img_width,int img_height,int img_channels,uint8* output){

    uint8 enable_auto_contrast = auto_contrast_default_para.enable_auto_contrast;
    float auto_black_percentage = auto_contrast_default_para.black_percentage / 100.0;
    float auto_white_percentage = auto_contrast_default_para.white_percentage / 100.0;

    uint8 auto_black_min = auto_contrast_default_para.auto_black_min;
    uint8 auto_black_max = auto_contrast_default_para.auto_black_max;
    uint8 auto_white_prc_target = auto_contrast_default_para.auto_white_prc_target;
    uint8 low_contrast = auto_contrast_default_para.low_contrast;
    uint8 high_contrast = auto_contrast_default_para.high_contrast;

    uint32 auto_contrast_offset = 0;
    float auto_contrast_gain = 0;


    if(enable_auto_contrast){

        uint32 pixel_count = 0;
        uint32 pixel_thr;
        int i =0;
        uint8 auto_white_index;
        uint8 auto_black_index;
        float white_gain;

        pixel_count = cumu_hist[HISTOGRAM_SIZE-1];
        //printf("pixel_count:%d\n",pixel_count);
        pixel_thr = (uint32)(auto_white_percentage * pixel_count);
        i = HISTOGRAM_SIZE - 1;
        while ( cumu_hist[i] >= pixel_thr && i >1 ) {
            i--;
        }
        auto_white_index = i;
        auto_white_index = ( auto_white_index <= HISTOGRAM_SIZE / 2 ) ? HISTOGRAM_SIZE / 2 : auto_white_index;
        white_gain = HISTOGRAM_SIZE * auto_white_prc_target / 100.0 / auto_white_index;
        printf("white_gain:%f\n",white_gain);
        float max_gain_clip;
        max_gain_clip = HISTOGRAM_SIZE * 99 / 100.0 / auto_white_index;

        pixel_thr = (uint32)(auto_black_percentage * pixel_count);
        i = auto_white_index;
        while (cumu_hist[i] >= pixel_thr && i >1 ) {
            i--;
        }
        auto_black_index = i;
        //printf("auto_black_index:%d\n",auto_black_index);
        float contrast = 1.0*auto_white_index / auto_black_index;
        printf("contrast:%f\n",contrast);

        auto_black_index = CLIP3(auto_black_index,auto_black_min,auto_black_max);
        auto_contrast_offset = auto_black_index;
        printf("auto_contrast_offset:%d\n",auto_contrast_offset);
#if 1
        //uint32 max_gain_contrast = HISTOGRAM_SIZE <<8 / ( ( HISTOGRAM_SIZE - auto_contrast_offset ) ? ( HISTOGRAM_SIZE - auto_contrast_offset ) : 1 );
        float max_gain_contrast = 256.0 / ( ( HISTOGRAM_SIZE - auto_contrast_offset ) ? ( HISTOGRAM_SIZE - auto_contrast_offset ) : 1 );

        float m, cy1 = 1.0, cy2 = 0, alpha = 0;
        uint8 cx1 = low_contrast,cx2 = high_contrast;
        m = ( cy1 - cy2 )/ ( cx1 - cx2 );
        alpha = m * ( contrast - cx1 ) + cy1;
        alpha = alpha < 0 ? 0 : alpha;
        alpha = alpha > 1.0 ? 1.0 : alpha;
        printf("max_gain_contrast:%f,max_gain_clip:%f,alpha:%f\n",max_gain_contrast,max_gain_clip,alpha);
        max_gain_clip = ( alpha * max_gain_contrast ) + ( ( 1.0 - alpha ) * max_gain_clip );

        max_gain_clip = CLIP3(max_gain_clip,0.0,16.0);
        printf("max_gain_clip:%f\n",max_gain_clip);
#endif
        //auto_contrast_gain = HISTOGRAM_SIZE / ( ( HISTOGRAM_SIZE - auto_contrast_offset ) ? ( HISTOGRAM_SIZE - auto_contrast_offset ) : 1 );
        auto_contrast_gain = 256.0 / ( ( HISTOGRAM_SIZE - auto_contrast_offset ) ? ( HISTOGRAM_SIZE - auto_contrast_offset ) : 1 );


        auto_contrast_gain = CLIP3(auto_contrast_gain,white_gain,max_gain_clip);
        printf("auto_contrast_gain:%f,auto_contrast_offset:%d\n",auto_contrast_gain,auto_contrast_offset);
    }
    else{
        auto_contrast_gain = 1.0;
        auto_contrast_offset = 0;
    }

        //out = gain*(in -offset);


#if 1
        auto_contrast_offset = 0;
        for(int y = 0;y < img_height;y++){
            for(int x =0;x < img_width;x++){
                uint8 input_pixel_r = input[img_channels*(y*img_width+x)];
                uint8 input_pixel_g = input[img_channels*(y*img_width+x)+1];
                uint8 input_pixel_b = input[img_channels*(y*img_width+x)+2];
                uint8 temp_r = 0,temp_g =0,temp_b=0;
                temp_r = CLIP3(input_pixel_r-auto_contrast_offset,0,255);
                temp_g = CLIP3(input_pixel_g-auto_contrast_offset,0,255);
                temp_b = CLIP3(input_pixel_b-auto_contrast_offset,0,255);

                output[img_channels*(y*img_width+x)] = gamma_045_table[temp_r];
                output[img_channels*(y*img_width+x)+1] = gamma_045_table[temp_g];
                output[img_channels*(y*img_width+x)+2] = gamma_045_table[temp_b];

                uint8 temp_out_r = output[img_channels*(y*img_width+x)];
                uint8 temp_out_g = output[img_channels*(y*img_width+x)+1];
                uint8 temp_out_b = output[img_channels*(y*img_width+x)+2];

                output[img_channels*(y*img_width+x)] = CLIP3(floor(temp_out_r * auto_contrast_gain),0,255);
                output[img_channels*(y*img_width+x)+1] = CLIP3(floor(temp_out_g * auto_contrast_gain),0,255);
                output[img_channels*(y*img_width+x)+2] = CLIP3(floor(temp_out_b * auto_contrast_gain),0,255);
                //uint8 out_r = CLIP3(floor(temp_r * auto_contrast_gain),0,255);
                //uint8 out_g = CLIP3(floor(temp_g * auto_contrast_gain),0,255);
                //uint8 out_b = CLIP3(floor(temp_b * auto_contrast_gain),0,255);

                //output[img_channels*(y*img_width+x)] = out_r;
                //output[img_channels*(y*img_width+x)+1] = out_g;
                //output[img_channels*(y*img_width+x)+2] = out_b;
            }
        }
#endif

}
static auto_contrast_para auto_contrast_default_para = {
        1,   // enable_auto_contrast
        10,  // black_percentage//越大，contrast越低，contrast gain越大，越接近max_gain_contrast（越大，auto_black_index越大，max_gain_contrast越大）
        99, // white_percentage//越大，contrast越大，contrast gain越小，越接近max_gain_clip
        1,  // auto_black_min //offset clip的下限
        50, // auto_black_max //offset clip的上限
        90, // auto_white_prc_target //gain,clip的下限
        20, // low contrast
        50, // high contrast
};
int main() {
    /*
    if (argc < 3) {
        printf("usage: %s   input_image  output_image \n ", argv[0]);
        printf("eg: %s   d:\\input.jpg   d:\\ouput.jpg\n ", argv[0]);
        return 0;
    }
    char* szfile = argv[1];
    char* out_img_name = argv[2];
     */
    char* szfile = "D:\\clion_project\\auto_contrast\\test_img2.bmp";
    int Width = 0;
    int Height = 0;
    int Channels = 0;
    char* out_img_name = "D:\\clion_project\\auto_contrast\\result\\022.bmp";

    Mat src_img;
    src_img = imread(szfile);
    Width = src_img.cols;
    Height = src_img.rows;
    Channels = src_img.channels();
    //printf("ww_%d,hh_%d,CC_%d",Width,Height,Channels);
    //imshow("src_image",src_img);
    //waitKey();

    if ((Channels != 0) && (Width != 0) && (Height != 0)) {
        uint8* inputImg = (uint8 *) malloc(Width * Channels * Height * sizeof(uint8));
        memcpy(inputImg, src_img.data, Width * Channels * Height*sizeof(uint8));

        //printf("ooookkkk!\n");

        //rgb2gray //I=R*0.2989+G*0.5871+0.1140*B;
        uint8* gray = (uint8 *) malloc(Width * Height * sizeof(uint8));
        if (gray) {
            memset(gray, 0, Width * Height * sizeof(uint8));
        }
        uint8* output = (uint8 *) malloc(Width * Channels * Height * sizeof(uint8));
        if (output) {
            memset(output, 0, Width * Channels * Height * sizeof(uint8));
        }
        uint8* linear_in = (uint8 *) malloc(Width * Channels * Height * sizeof(uint8));
        if (linear_in) {
            memset(linear_in, 0, Width * Channels * Height * sizeof(uint8));
        }
        apply_invert_gamma(inputImg,linear_in,Width,Height,Channels);
        for(int y = 0;y < Height;y++){
            for(int x =0;x < Width;x++){

                uint8 temp_r = inputImg[Channels*(y*Width+x)];
                uint8 temp_g = inputImg[Channels*(y*Width+x)+1];
                uint8 temp_b = inputImg[Channels*(y*Width+x)+2];

                uint8  temp_gray = floor(0.2989*temp_r+0.5871*temp_g+0.1140*temp_b);
                gray[y*Width+x] = CLIP3(temp_gray,0,255);
            }
        }

        uint32 hist[HISTOGRAM_SIZE] = {0};
        hist_stats(gray,Width,Height,hist);

        uint32 cumu_hist[HISTOGRAM_SIZE]={0};
        cumu_hist[0] = hist[0];
        for ( int i = 1; i < HISTOGRAM_SIZE; i++ ) {
            cumu_hist[i] = cumu_hist[i - 1] + hist[i];
        }
        auto_contrast(linear_in,cumu_hist,auto_contrast_default_para,Width,Height,Channels,output);

        Mat dst_image = Mat(1080,1920,CV_8UC3,(uint8*)output);
        //imshow("dst_image",dst_image);
        //waitKey();
        imwrite(out_img_name,dst_image);
        free(linear_in);
        free(output);
        free(gray);
        free(inputImg);
    }
    else{
        printf("load: %s fail!\n", szfile);
    }
    return 0;
}
