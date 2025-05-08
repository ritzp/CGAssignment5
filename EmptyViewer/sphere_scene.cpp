//
//  sphere_scene.c
//  Rasterizer
//
//

#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>
#include <limits>

using namespace std;

const int WIDTH = 512;
const int HEIGHT = 512;

struct Vertex {
public:
    float x, y, z, w = 1.0f;
};

struct Vector3 {
public:
    float x, y, z;
};

int gNumVertices;
int gNumTriangles;
std::vector<Vertex> gVertices;
int* gIndexBuffer;
float depthBuffer[WIDTH][HEIGHT];
std::vector<float> OutputImage;

// Matrix Multiplication Helper
Vertex multiply(const float matrix[4][4], const Vertex& v) {
    Vertex result = { 0, 0, 0, 0 };
    result.x = matrix[0][0] * v.x + matrix[0][1] * v.y + matrix[0][2] * v.z + matrix[0][3] * v.w;
    result.y = matrix[1][0] * v.x + matrix[1][1] * v.y + matrix[1][2] * v.z + matrix[1][3] * v.w;
    result.z = matrix[2][0] * v.x + matrix[2][1] * v.y + matrix[2][2] * v.z + matrix[2][3] * v.w;
    result.w = matrix[3][0] * v.x + matrix[3][1] * v.y + matrix[3][2] * v.z + matrix[3][3] * v.w;
    return result;
}

Vertex modelingTransform(Vertex _v, float sx, float sy, float sz, float tx, float ty, float tz) {
    float mm[4][4] = {
        {sx, 0, 0, tx},
        {0, sy, 0, ty},
        {0, 0, sz, tz},
        { 0, 0, 0, 1 }
    };
    return multiply(mm, _v);
}

Vertex cameraTransform(Vertex _v, Vector3 u, Vector3 v, Vector3 w, Vector3 e) {
    float mc[4][4] = {
        {u.x, v.x, w.x, e.x},
        {u.y, v.y, w.y, e.y},
        {u.z, v.z, w.z, e.z},
        {0, 0, 0, 1}
    };
    return multiply(mc, _v);
}

Vertex projectionTransform(Vertex _v, float l, float r, float t, float b, float n, float f) {
    float mp[4][4] = {
        {2 * n / (r - l), 0, (l + r) / (l - r), 0},
        {0, 2 * n / (t - b), (b + t) / (b - t), 0},
        {0, 0, (f + n) / (n - f), (2 * f * n) / (f - n)},
        {0, 0, 1, 0}
    };
    _v = multiply(mp, _v);
    if (_v.w != 0) {
        _v.x /= _v.w;
        _v.y /= _v.w;
        _v.z /= _v.w;
        _v.w = 1.0f;
    }
    return _v;
}

Vertex viewportTransform(Vertex v, float width, float height) {
    v.x = (v.x + 1) * width * 0.5f;
    v.y = (v.y + 1) * height * 0.5f;
    return v;
}

void transform_and_rasterize() {
    std::fill(&depthBuffer[0][0], &depthBuffer[0][0] + WIDTH * HEIGHT, std::numeric_limits<float>::infinity());
    OutputImage.resize(WIDTH * HEIGHT * 3, 0.0f);

    for (int i = 0; i < gNumTriangles; ++i) {
        int idx0 = gIndexBuffer[3 * i + 0];
        int idx1 = gIndexBuffer[3 * i + 1];
        int idx2 = gIndexBuffer[3 * i + 2];

        Vertex v0 = gVertices[idx0];
        Vertex v1 = gVertices[idx1];
        Vertex v2 = gVertices[idx2];

        Vector3 u = { 1, 0, 0 };
        Vector3 v = { 0, 1, 0 };
        Vector3 w = { 0, 0, 1 };
        Vector3 e = { 0, 0, 0 };

        // Transform
        v0 = modelingTransform(v0, 2, 2, 2, 0, 0, -7);
        v1 = modelingTransform(v1, 2, 2, 2, 0, 0, -7);
        v2 = modelingTransform(v2, 2, 2, 2, 0, 0, -7);

        v0 = cameraTransform(v0, u, v, w, e);
        v1 = cameraTransform(v1, u, v, w, e);
        v2 = cameraTransform(v2, u, v, w, e);

        v0 = projectionTransform(v0, -0.1, 0.1, 0.1, -0.1, -0.1, -1000);
        v1 = projectionTransform(v1, -0.1, 0.1, 0.1, -0.1, -0.1, -1000);
        v2 = projectionTransform(v2, -0.1, 0.1, 0.1, -0.1, -0.1, -1000);

        v0 = viewportTransform(v0, WIDTH, HEIGHT);
        v1 = viewportTransform(v1, WIDTH, HEIGHT);
        v2 = viewportTransform(v2, WIDTH, HEIGHT);

        // Rasterization
        int minX = (int)floorf(fmin(fmin(v0.x, v1.x), v2.x));
        int maxX = (int)ceilf(fmax(fmax(v0.x, v1.x), v2.x));
        int minY = (int)floorf(fmin(fmin(v0.y, v1.y), v2.y));
        int maxY = (int)ceilf(fmax(fmax(v0.y, v1.y), v2.y));

        minX = std::max(minX, 0);
        maxX = std::min(maxX, 511);
        minY = std::max(minY, 0);
        maxY = std::min(maxY, 511);

        float denom = (v1.y - v2.y) * (v0.x - v2.x) + (v2.x - v1.x) * (v0.y - v2.y);
        if (denom == 0) continue;

        for (int y = minY; y <= maxY; ++y) {
            for (int x = minX; x <= maxX; ++x) {
                float px = (float)x + 0.5f;
                float py = (float)y + 0.5f;

                float w0 = ((v1.y - v2.y) * (px - v2.x) + (v2.x - v1.x) * (py - v2.y)) / denom;
                float w1 = ((v2.y - v0.y) * (px - v2.x) + (v0.x - v2.x) * (py - v2.y)) / denom;
                float w2 = 1.0f - w0 - w1;

                if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
                    float z = w0 * v0.z + w1 * v1.z + w2 * v2.z;

                    if (z < depthBuffer[x][y]) {
                        depthBuffer[x][y] = z;

                        int index = (y * WIDTH + x) * 3;
                        OutputImage[index + 0] = 1.0f; // R
                        OutputImage[index + 1] = 1.0f; // G
                        OutputImage[index + 2] = 1.0f; // B
                    }
                }
            }
        }
    }
}

void create_scene()
{
    int width   = 32;
    int height  = 16;
    
    float theta, phi;
    int t;
    
    gNumVertices    = (height - 2) * width + 2;
    gNumTriangles   = (height - 2) * (width - 1) * 2;
    
    // TODO: Allocate an array for gNumVertices vertices.
    gVertices.resize(gNumVertices);

    gIndexBuffer    = new int[3*gNumTriangles];
    
    t = 0;
    for (int j = 1; j < height-1; ++j)
    {
        for (int i = 0; i < width; ++i)
        {
            theta = (float) j / (height-1) * M_PI;
            phi   = (float) i / (width-1)  * M_PI * 2;
            
            float   x   = sinf(theta) * cosf(phi);
            float   y   = cosf(theta);
            float   z   = -sinf(theta) * sinf(phi);
            
            // TODO: Set vertex t in the vertex array to {x, y, z}.
            gVertices[t] = { x, y, z };
            
            t++;
        }
    }
    
    // TODO: Set vertex t in the vertex array to {0, 1, 0}.
    gVertices[t] = { 0, 1, 0 };
    
    t++;
    
    // TODO: Set vertex t in the vertex array to {0, -1, 0}.
    gVertices[t] = { 0, -1, 0 };
    
    t++;
    
    t = 0;
    for (int j = 0; j < height-3; ++j)
    {
        for (int i = 0; i < width-1; ++i)
        {
            gIndexBuffer[t++] = j*width + i;
            gIndexBuffer[t++] = (j+1)*width + (i+1);
            gIndexBuffer[t++] = j*width + (i+1);
            gIndexBuffer[t++] = j*width + i;
            gIndexBuffer[t++] = (j+1)*width + i;
            gIndexBuffer[t++] = (j+1)*width + (i+1);
        }
    }
    for (int i = 0; i < width-1; ++i)
    {
        gIndexBuffer[t++] = (height-2)*width;
        gIndexBuffer[t++] = i;
        gIndexBuffer[t++] = i + 1;
        gIndexBuffer[t++] = (height-2)*width + 1;
        gIndexBuffer[t++] = (height-3)*width + (i+1);
        gIndexBuffer[t++] = (height-3)*width + i;
    }
    
    // The index buffer has now been generated. Here's how to use to determine
    // the vertices of a triangle. Suppose you want to determine the vertices
    // of triangle i, with 0 <= i < gNumTriangles. Define:
    //
    // k0 = gIndexBuffer[3*i + 0]
    // k1 = gIndexBuffer[3*i + 1]
    // k2 = gIndexBuffer[3*i + 2]
    //
    // Now, the vertices of triangle i are at positions k0, k1, and k2 (in that
    // order) in the vertex array (which you should allocate yourself at line
    // 27).
    //
    // Note that this assumes 0-based indexing of arrays (as used in C/C++,
    // Java, etc.) If your language uses 1-based indexing, you will have to
    // add 1 to k0, k1, and k2.
}



//
// main.cpp
// Output Image
//
//

#include <Windows.h>
#include <iostream>
#include <GL/glew.h>
#include <GL/GL.h>
#include <GL/freeglut.h>

#define GLFW_INCLUDE_GLU
#define GLFW_DLL
#include <GLFW/glfw3.h>
#include <vector>

#define GLM_SWIZZLE
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>

void render()
{
    //Create our image. We don't want to do this in 
    //the main loop since this may be too slow and we 
    //want a responsive display of our beautiful image.
    //Instead we draw to another buffer and copy this to the 
    //framebuffer using glDrawPixels(...) every refresh
    create_scene();
    transform_and_rasterize();
}

void resize_callback(GLFWwindow*, int nw, int nh)
{
    //This is called in response to the window resizing.
    //The new width and height are passed in so we make 
    //any necessary changes:
    //WIDTH = nw;
    //HEIGHT = nh;
    //Tell the viewport to use all of our screen estate
    glViewport(0, 0, WIDTH, HEIGHT);

    //This is not necessary, we're just working in 2d so
    //why not let our spaces reflect it?
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glOrtho(0.0, static_cast<double>(WIDTH)
        , 0.0, static_cast<double>(HEIGHT)
        , 1.0, -1.0);

    //Reserve memory for our render so that we don't do 
    //excessive allocations and render the image
    OutputImage.reserve(WIDTH * HEIGHT * 3);
    render();
}


int main(int argc, char* argv[])
{
    // -------------------------------------------------
    // Initialize Window
    // -------------------------------------------------

    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(WIDTH, HEIGHT, "OpenGL Viewer", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    //We have an opengl context now. Everything from here on out 
    //is just managing our window or opengl directly.

    //Tell the opengl state machine we don't want it to make 
    //any assumptions about how pixels are aligned in memory 
    //during transfers between host and device (like glDrawPixels(...) )
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);

    //We call our resize function once to set everything up initially
    //after registering it as a callback with glfw
    glfwSetFramebufferSizeCallback(window, resize_callback);
    resize_callback(NULL, WIDTH, HEIGHT);

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        //Clear the screen
        glClear(GL_COLOR_BUFFER_BIT);

        // -------------------------------------------------------------
        //Rendering begins!
        glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_FLOAT, &OutputImage[0]);
        //and ends.
        // -------------------------------------------------------------

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();

        //Close when the user hits 'q' or escape
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS
            || glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(window, GL_TRUE);
        }
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
