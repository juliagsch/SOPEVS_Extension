#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

using namespace std;
using json = nlohmann::json;

const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    void main() {
        gl_Position = projection * view * model * vec4(aPos, 1.0);
    }
)";

const char* fragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    uniform vec3 color;
    void main() {
        FragColor = vec4(color, 1.0);
    }
)";

struct PairInfo {
    int start;
    int count;
    glm::vec3 color;
};

vector<glm::vec3> all_vertices;
vector<PairInfo> pairs;
GLuint shaderProgram, VAO, VBO;

void parseJSON(const string& input) {
    string json_str = input;
    replace(json_str.begin(), json_str.end(), '\'', '\"');
    json data;
    try {
        data = json::parse(json_str);
    } catch (json::parse_error& e) {
        cerr << "JSON parse error: " << e.what() << endl;
        exit(1);
    }

    vector<pair<string, json>> meshes;
    for (auto& el : data.items()) meshes.emplace_back(el.key(), el.value());
    
    // Explicitly specify the lambda parameter types instead of using 'auto'
    sort(meshes.begin(), meshes.end(), [](const pair<string, json>& a, const pair<string, json>& b) {
        return a.first < b.first;
    });
    
    for (size_t i = 0; i < meshes.size(); i += 2) {
        if (i + 1 >= meshes.size()) break;

        auto& mesh1 = meshes[i].second;
        auto& mesh2 = meshes[i + 1].second;

        int start = all_vertices.size();
        for (auto& point : mesh1["original_coordinates"]) {
            all_vertices.emplace_back(point[0].get<float>(), point[1].get<float>(), point[2].get<float>());
        }
        for (auto& point : mesh2["original_coordinates"]) {
            all_vertices.emplace_back(point[0].get<float>(), point[1].get<float>(), point[2].get<float>());
        }

        float avg = (mesh1["average_radiance"].get<float>() + mesh2["average_radiance"].get<float>()) / 2.0f;
        pairs.push_back({start, 6, glm::vec3(avg / 100.0f)});
    }
}

void setupBuffers() {
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, all_vertices.size() * sizeof(glm::vec3), all_vertices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void compileShaders() {
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

int main() {
    if (!glfwInit()) {
        cerr << "Failed to initialize GLFW" << endl;
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(800, 600, "Mesh Pairs", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        cerr << "Failed to initialize GLEW" << endl;
        return -1;
    }

    glEnable(GL_DEPTH_TEST);

    string input_json = "{'Mesh_1': {'average_radiance': 100.03492735694, 'original_coordinates': [[0.11370011238620566, 52.1985753079607, 4.266192772977321], [0.1136872797592018, 52.19857096841214, 4.266192772977321], [0.11369342226548204, 52.198564144506975, 3.76889409467354]], 'centroid': [0.1136936, 52.19857014, 4.10042655], 'mesh_index': 0}, 'Mesh_2': {'average_radiance': 100.03492896355068, 'original_coordinates': [[0.11370011238620566, 52.1985753079607, 4.266192772977321], [0.11369342226548204, 52.198564144506975, 3.76889409467354], [0.1137062548924859, 52.198568484055535, 3.76889409467354]], 'centroid': [0.11369993, 52.19856931, 3.93466032], 'mesh_index': 0}, 'Mesh_3': {'average_radiance': 100.03493358628137, 'original_coordinates': [[0.1137062548924859, 52.198568484055535, 3.76889409467354], [0.11369342226548204, 52.198564144506975, 3.76889409467354], [0.1136936556405145, 52.198563885243246, 3.75]], 'centroid': [0.11369778, 52.1985655, 3.76259606], 'mesh_index': 1}, 'Mesh_4': {'average_radiance': 100.03493225241574, 'original_coordinates': [[0.1137062548924859, 52.198568484055535, 3.76889409467354], [0.1136936556405145, 52.198563885243246, 3.75], [0.11370648826751835, 52.198568224791806, 3.75]], 'centroid': [0.11370213, 52.19856686, 3.75629803], 'mesh_index': 1}, 'Mesh_5': {'average_radiance': 100.03492253709966, 'original_coordinates': [[0.11368113725292155, 52.19857779231731, 4.763491451281101], [0.11366830462591769, 52.19857345276875, 4.763491451281101], [0.11367444713219793, 52.19856662886358, 4.266192772977321]], 'centroid': [0.11367463, 52.19857262, 4.59772523], 'mesh_index': 2}, 'Mesh_6': {'average_radiance': 100.0349241437383, 'original_coordinates': [[0.11368113725292155, 52.19857779231731, 4.763491451281101], [0.11367444713219793, 52.19856662886358, 4.266192772977321], [0.1136872797592018, 52.19857096841214, 4.266192772977321]], 'centroid': [0.11368095, 52.1985718, 4.431959], 'mesh_index': 2}, 'Mesh_7': {'average_radiance': 100.03493170694773, 'original_coordinates': [[0.1136872797592018, 52.19857096841214, 4.266192772977321], [0.11367444713219793, 52.19856662886358, 4.266192772977321], [0.11368058963847819, 52.198559804958414, 3.76889409467354]], 'centroid': [0.11368077, 52.1985658, 4.10042655], 'mesh_index': 3}, 'Mesh_8': {'average_radiance': 100.03493331353764, 'original_coordinates': [[0.1136872797592018, 52.19857096841214, 4.266192772977321], [0.11368058963847819, 52.198559804958414, 3.76889409467354], [0.11369342226548204, 52.198564144506975, 3.76889409467354]], 'centroid': [0.1136871, 52.19856497, 3.93466032], 'mesh_index': 3}, 'Mesh_9': {'average_radiance': 100.03493793626743, 'original_coordinates': [[0.11369342226548204, 52.198564144506975, 3.76889409467354], [0.11368058963847819, 52.198559804958414, 3.76889409467354], [0.11368082301351064, 52.198559545694685, 3.75]], 'centroid': [0.11368494, 52.19856117, 3.76259606], 'mesh_index': 4}, 'Mesh_10': {'average_radiance': 100.0349366024038, 'original_coordinates': [[0.11369342226548204, 52.198564144506975, 3.76889409467354], [0.11368082301351064, 52.198559545694685, 3.75], [0.1136936556405145, 52.198563885243246, 3.75]], 'centroid': [0.1136893, 52.19856253, 3.75629803], 'mesh_index': 4}}";
    parseJSON(input_json);
    setupBuffers();
    compileShaders();

    glm::mat4 view = glm::lookAt(glm::vec3(0.1f, 52.1985f, 10.0f), glm::vec3(0.1f, 52.1985f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
    glm::mat4 model = glm::mat4(1.0f);

    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(shaderProgram);

        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, &model[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, &view[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, &projection[0][0]);

        glBindVertexArray(VAO);
        for (auto& pair : pairs) {
            glUniform3fv(glGetUniformLocation(shaderProgram, "color"), 1, &pair.color[0]);
            glDrawArrays(GL_TRIANGLES, pair.start, pair.count);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glfwTerminate();
    return 0;
}