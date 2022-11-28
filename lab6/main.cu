#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <random>

// resolution
	int w = 1024, h = 648;
// camera parametrs 
	float cam_x = -1.5, cam_y = -1.5, cam_z = 1.0;
	float cam_dx = 0.0, cam_dy = 0.0, cam_dz = 0.0;
	float cam_yaw = 0.0, cam_pitch = 0.0;
	float cam_dyaw = 0.0, cam_dpitch = 0.0;
	float cam_speed = 0.05;
// bounding box parametrs
	const float offset = 15.0;
// texture
	GLuint textures[2];				// Массив из текстурных номеров
	GLuint* particle_texture = &textures[0];
	GLuint* floor_texture = &textures[1];
	const uint floor_texture_size = 128;
	GLuint pbo;						// Номер буфера
	cudaGraphicsResource *pbo_res;
// particles parametrs
	GLUquadric* quadratic;
	struct particle {
		float x;
		float y;
		float z;
		float dx = 0;
		float dy = 0;
		float dz = 0;
		float q = 0.1;
	};
	particle projectile{10000.0, 10000.0, 10000.0, 0.0, 0.0, 0.0, 2.0};
	const float projectile_speed = 0.5;
	particle cam_particle{cam_x, cam_y, cam_z, cam_dx, cam_dy, cam_dz, 1.0};
	particle* particles;
	uint num_particles = 10;
// physics parametrs
	const float e = 1e-3;
	const float dt = 0.005;
	const float W = 0.9999;
	const float K = 50.0;

void reshape(int w_new, int h_new) {
	w = w_new;
	h = h_new;
	glViewport(0, 0, w, h);                                     // Сброс текущей области вывода
	glMatrixMode(GL_PROJECTION);                                // Выбор матрицы проекций
	glLoadIdentity();                                           // Сброс матрицы проекции
}

void keys(unsigned char key, int x, int y) {	// Обработка кнопок
	switch (key) {
		case 'w':                 // "W" Движение вперед
			cam_dx += cos(cam_yaw) * cos(cam_pitch) * cam_speed;
			cam_dy += sin(cam_yaw) * cos(cam_pitch) * cam_speed;
			cam_dz += sin(cam_pitch) * cam_speed;
		break;
		case 's':                 // "S" Назад
			cam_dx += -cos(cam_yaw) * cos(cam_pitch) * cam_speed;
			cam_dy += -sin(cam_yaw) * cos(cam_pitch) * cam_speed;
			cam_dz += -sin(cam_pitch) * cam_speed;
		break;
		case 'a':                 // "A" Влево
			cam_dx += -sin(cam_yaw) * cam_speed;
			cam_dy += cos(cam_yaw) * cam_speed;
			break;
		case 'd':                 // "D" Вправо
			cam_dx += sin(cam_yaw) * cam_speed;
			cam_dy += -cos(cam_yaw) * cam_speed;
		break;
		case 27:
			cudaGraphicsUnregisterResource(pbo_res);
			glDeleteTextures(2, textures);
			glDeleteBuffers(1, &pbo);
			gluDeleteQuadric(quadratic);
			exit(0);
		break;
	}
}

void mouse(int x, int y) {
	static int x_prev = w / 2, y_prev = h / 2;
	float dx = 0.005 * (x - x_prev);
    float dy = 0.005 * (y - y_prev);
	cam_dyaw -= dx;
    cam_dpitch -= dy;
	x_prev = x;
	y_prev = y;

	// Перемещаем указатель мышки в центр, когда он достиг границы
	if ((x < 20) || (y < 20) || (x > w - 20) || (y > h - 20)) {
		glutWarpPointer(w / 2, h / 2);
		x_prev = w / 2;
		y_prev = h / 2;
    }
}

void mouse_click(int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON or button == GLUT_RIGHT_BUTTON){
		projectile.x = cam_x;
		projectile.y = cam_y;
		projectile.z = cam_z;
		projectile.dx = cos(cam_yaw) * cos(cam_pitch) * projectile_speed;
		projectile.dy = sin(cam_yaw) * cos(cam_pitch) * projectile_speed;
		projectile.dz = sin(cam_pitch) * projectile_speed;
	}
}

void particles_random_init(){
	particles = (particle*)malloc(sizeof(particle)*num_particles);

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dist(-offset+1, offset-1);
	for (int i = 0; i < num_particles; i++) {
		particles[i].x = dist(gen);
		particles[i].y = dist(gen);
		particles[i].z = dist(gen);
	}
}

void draw_bbox(){
	// Отрисовка каркаса куба				
	glLineWidth(2);								// Толщина линий				
	glColor3f(0.5f, 0.5f, 0.5f);				// Цвет линий
	glBegin(GL_LINES);							// Последующие пары вершин будут задавать линии
		glVertex3f(-offset, -offset, 0.0);
		glVertex3f(-offset, -offset, 2.0 * offset);

		glVertex3f(offset, -offset, 0.0);
		glVertex3f(offset, -offset, 2.0 * offset);

		glVertex3f(offset, offset, 0.0);
		glVertex3f(offset, offset, 2.0 * offset);

		glVertex3f(-offset, offset, 0.0);
		glVertex3f(-offset, offset, 2.0 * offset);
	glEnd();

	glBegin(GL_LINE_LOOP);						// Все последующие точки будут соеденены замкнутой линией
		glVertex3f(-offset, -offset, 0.0);
		glVertex3f(offset, -offset, 0.0);
		glVertex3f(offset, offset, 0.0);
		glVertex3f(-offset, offset, 0.0);
	glEnd();

	glBegin(GL_LINE_LOOP);
		glVertex3f(-offset, -offset, 2.0 * offset);
		glVertex3f(offset, -offset, 2.0 * offset);
		glVertex3f(offset, offset, 2.0 * offset);
		glVertex3f(-offset, offset, 2.0 * offset);
	glEnd();
}

void display() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	// Задаем "объектив камеры"
	gluPerspective(90.0f, (GLfloat)w/(GLfloat)h, 0.1f, 100.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// Задаем позицию и направление камеры
	gluLookAt(cam_x, cam_y, cam_z,
			  cam_x + cos(cam_yaw) * cos(cam_pitch),
			  cam_y + sin(cam_yaw) * cos(cam_pitch),
			  cam_z + sin(cam_pitch),
			  0.0f, 0.0f, 1.0f);

	draw_bbox();

	// printf("update\n");
	// printf("%f %f", cam_)
	glutSwapBuffers();
}

void update(){
	float v = sqrt(cam_dx*cam_dx+cam_dy*cam_dy+cam_dz*cam_dz);
	if (v > cam_speed) {		// Ограничение максимальной скорости
		cam_dx *= cam_speed / v;
		cam_dy *= cam_speed / v;
		cam_dz *= cam_speed / v;
	}
	cam_x += cam_dx; cam_dx *= 0.99;
	cam_y += cam_dy; cam_dy *= 0.99;
	cam_z += cam_dz; cam_dz *= 0.99;
	if (cam_z < 1.0) {			// Пол, ниже которого камера не может переместиться
		cam_z = 1.0;
		cam_dz = 0.0;
	}
	if (fabs(cam_dpitch) + fabs(cam_dyaw) > 0.0001) {	// Вращение камеры
		cam_yaw += cam_dyaw;
		cam_pitch += cam_dpitch;
		cam_pitch = min(M_PI / 2.0 - 0.0001, max(-M_PI / 2.0 + 0.0001, cam_pitch));
		cam_dyaw = cam_dpitch = 0.0;
	}

	glutPostRedisplay();
}

int main(int argc, char **argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(w, h);
	glutCreateWindow("OpenGL");

	glutIdleFunc(update);
	glutDisplayFunc(display);
	glutKeyboardFunc(keys);
	glutPassiveMotionFunc(mouse);
	glutMouseFunc(mouse_click);
	glutReshapeFunc(reshape);

	glutSetCursor(GLUT_CURSOR_NONE);	// Скрываем курсор мышки

	// load particle texture
		int texture_w, texture_h;
		FILE *fp = fopen("in.data", "rb");
		fread(&texture_w, sizeof(int), 1, fp);
		fread(&texture_h, sizeof(int), 1, fp);
		uchar4* data = (uchar4*)malloc(sizeof(uchar4)*texture_w*texture_h);
		fread(data, sizeof(uchar4), texture_w*texture_h, fp);
		fclose(fp);

		//
		glGenTextures(2, textures);
		glBindTexture(GL_TEXTURE_2D, *particle_texture);
		glTexImage2D(GL_TEXTURE_2D, 0, 3, (GLsizei)texture_w, (GLsizei)texture_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, (void*)data);
		free(data);
		// если полигон, на который наносим текстуру, меньше текстуры
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); //GL_LINEAR);	// Интерполяция
		// если больше
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); //GL_LINEAR);

		glBindTexture(GL_TEXTURE_2D, *floor_texture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);	// Интерполяция 
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);	// Интерполяция	
	
	// init quadratic
		quadratic = gluNewQuadric();
		gluQuadricTexture(quadratic, GL_TRUE);

	// GL parametrs setup
		glEnable(GL_TEXTURE_2D);                             // Разрешить наложение текстуры
		glShadeModel(GL_SMOOTH);                             // Разрешение сглаженного закрашивания
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);                // Черный фон
		glClearDepth(1.0f);                                  // Установка буфера глубины
		glDepthFunc(GL_LEQUAL);                              // Тип теста глубины. 
		glEnable(GL_DEPTH_TEST);                			 // Включаем тест глубины
		glEnable(GL_CULL_FACE);                 			 // Режим при котором, тектуры накладываются только с одной стороны

	glewInit();						
	// floor pixel buffer object init and setup
		glGenBuffers(1, &pbo);								// Получаем номер буфера
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);			// Делаем его активным
		glBufferData(GL_PIXEL_UNPACK_BUFFER, floor_texture_size*floor_texture_size*sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);	// Задаем размер буфера
		cudaGraphicsGLRegisterBuffer(&pbo_res, pbo, cudaGraphicsMapFlagsWriteDiscard);				// Регистрируем буфер для использования его памяти в CUDA
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);			// Деактивируем буфер

	// init scene
		particles_random_init();

	glutMainLoop();
}