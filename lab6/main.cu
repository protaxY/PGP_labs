#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <random>

#define sqr3(x) ((x)*(x)*(x))
#define sqr(x) ((x)*(x))

#define CSC(call)																							\
do {																										\
	cudaError_t status = call;																				\
	if (status != cudaSuccess) {																			\
		fprintf(stderr, "ERROR is %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status));	\
		exit(0);																							\
	}																										\
} while(0)	

// resolution
	int w = 1024, h = 648;
// bounding box parametrs
	const float offset = 15.0;
// camera parametrs 
	float cam_x = -1.5*offset, cam_y = 0, cam_z = 1.8*offset;
	float cam_dx = 0.0, cam_dy = 0.0, cam_dz = 0.0;
	float cam_yaw = 0.0, cam_pitch = 0.0;
	float cam_dyaw = 0.0, cam_dpitch = 0.0;
	float cam_speed = 0.1;
// texture
	GLuint textures[2];				// Массив из текстурных номеров
	GLuint* particle_texture = &textures[0];
	GLuint* floor_texture = &textures[1];
	const uint floor_texture_size = 128;
	GLuint pbo;						// Номер буфера
	cudaGraphicsResource* pbo_res;
// particles parametrs
	GLUquadric* quadratic;
	struct particle {
		float x;
		float y;
		float z;
		float dx;
		float dy;
		float dz;
		float q;
		float angle;
		float spin;
	};
	particle projectile{10000.0, 10000.0, 10000.0, 0.0, 0.0, 0.0, 1000.0, 0};
	const float projectile_speed = 200.0;
	particle cam_particle{cam_x, cam_y, cam_z, cam_dx, cam_dy, cam_dz, 10.0, 0};
	particle* particles;
	particle* dev_particles;
	uint num_particles = 200;
// physics parametrs
	const float e = 1e-3;
	const float dt = 0.005;
	const float W = 0.99;
	const float K = 50.0;
	const float g = 15.0;

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
			cudaFree(dev_particles);
			free(particles);

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
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dist(-offset+1, offset-1);
	for (uint i = 0; i < num_particles; i++) {
		particles[i].x = dist(gen);
		particles[i].y = dist(gen);
		particles[i].z = abs(dist(gen))+1;

		particles[i].dx = 0;
		particles[i].dy = 0;
		particles[i].dz = 0;

		particles[i].q = 0.5;

		particles[i].angle = 0;
		particles[i].spin = 100*dist(gen);
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

void draw_particles(){
	glBindTexture(GL_TEXTURE_2D, *particle_texture);
	for (uint i = 0; i < num_particles; ++i){
		glPushMatrix();
			glTranslatef(particles[i].x, particles[i].y, particles[i].z);	// Задаем координаты центра сферы
			glRotatef(particles[i].angle, 0.0, 0.0, 1.0);
			gluSphere(quadratic, 0.5f, 2*4, 4);
		glPopMatrix();
	}
}

void draw_projectile(){
	glBindTexture(GL_TEXTURE_2D, 0);
	glPushMatrix();
		glTranslatef(projectile.x, projectile.y, projectile.z);	// Задаем координаты центра сферы
		gluSphere(quadratic, 1.0f, 2*4, 4);
	glPopMatrix();
}

void draw_floor(){
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);	// Делаем активным буфер с номером vbo
	glBindTexture(GL_TEXTURE_2D, *floor_texture);	// Делаем активной вторую текстуру
	glTexImage2D(GL_TEXTURE_2D, 0, 3, (GLsizei)floor_texture_size, (GLsizei)floor_texture_size, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL); 
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);	// Деактивируем буфер

	glBegin(GL_QUADS);			// Рисуем пол
		glTexCoord2f(0.0, 0.0);
		glVertex3f(-offset, -offset, 0.0);

		glTexCoord2f(1.0, 0.0);
		glVertex3f(offset, -offset, 0.0);

		glTexCoord2f(1.0, 1.0);
		glVertex3f(offset, offset, 0.0);

		glTexCoord2f(0.0, 1.0);
		glVertex3f(-offset, offset, 0.0);
	glEnd();

	glBindTexture(GL_TEXTURE_2D, 0);
}

__global__ void particle_kernel(particle* particles, uint num_particles, particle cam_particle, particle projectile, float K, float W, float g, float offset, float dt, float e) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	while (idx < num_particles){
		// замедление
			particles[idx].dx *= W;
			particles[idx].dy *= W;
			particles[idx].dz *= W;

		// Отталкивание от стен
			particles[idx].dx += 2*sqr(particles[idx].q)*K*(particles[idx].x-offset)/(sqr3(fabs(particles[idx].x-offset))+e)*dt;
			particles[idx].dx += 2*sqr(particles[idx].q)*K*(particles[idx].x+offset)/(sqr3(fabs(particles[idx].x+offset))+e)*dt;

			particles[idx].dy += 2*sqr(particles[idx].q)*K*(particles[idx].y-offset)/(sqr3(fabs(particles[idx].y-offset))+e)*dt;
			particles[idx].dy += 2*sqr(particles[idx].q)*K*(particles[idx].y+offset)/(sqr3(fabs(particles[idx].y+offset))+e)*dt;

			particles[idx].dz += 2*particles[idx].q*particles[idx].q*K*(particles[idx].z-2*offset)/(sqr3(fabs(particles[idx].z-2*offset))+e)*dt;
			particles[idx].dz += 2*particles[idx].q*particles[idx].q*K*(particles[idx].z)/(sqr3(fabs(particles[idx].z))+e)*dt;

		// Отталкивание от камеры
			float cam_r = sqrt(sqr(particles[idx].x-cam_particle.x)+sqr(particles[idx].y-cam_particle.y)+sqr(particles[idx].z-cam_particle.z));
			particles[idx].dx += cam_particle.q*particles[idx].q*K*(particles[idx].x-cam_particle.x)/(sqr3(cam_r)+e)*dt;
			particles[idx].dy += cam_particle.q*particles[idx].q*K*(particles[idx].y-cam_particle.y)/(sqr3(cam_r)+e)*dt;
			particles[idx].dz += cam_particle.q*particles[idx].q*K*(particles[idx].z-cam_particle.z)/(sqr3(cam_r)+e)*dt;

		// отталкивание от снаряда
			float projectile_r = sqrt(sqr(particles[idx].x-projectile.x)+sqr(particles[idx].y-projectile.y)+sqr(particles[idx].z-projectile.z));
			particles[idx].dx += projectile.q*particles[idx].q*K*(particles[idx].x-projectile.x)/(sqr3(projectile_r)+e)*dt;
			particles[idx].dy += projectile.q*particles[idx].q*K*(particles[idx].y-projectile.y)/(sqr3(projectile_r)+e)*dt;
			particles[idx].dz += projectile.q*particles[idx].q*K*(particles[idx].z-projectile.z)/(sqr3(projectile_r)+e)*dt;

		// отталкивание от остальных частиц
			for (uint i = 0; i < num_particles; ++i){
				if (idx == i)
					continue;
				
				float r = sqrt(sqr(particles[idx].x-particles[i].x)+sqr(particles[idx].y-particles[i].y)+sqr(particles[idx].z-particles[i].z));
				particles[idx].dx += particles[i].q*particles[idx].q*K*(particles[idx].x-particles[i].x)/(sqr3(r)+e)*dt;
				particles[idx].dy += particles[i].q*particles[idx].q*K*(particles[idx].y-particles[i].y)/(sqr3(r)+e)*dt;
				particles[idx].dz += particles[i].q*particles[idx].q*K*(particles[idx].z-particles[i].z)/(sqr3(r)+e)*dt;
			}

		// гравитация
			particles[idx].dz -= g*dt;

		// шаг по времени
			particles[idx].x += particles[idx].dx*dt;
			particles[idx].y += particles[idx].dy*dt;
			particles[idx].z += particles[idx].dz*dt;
			particles[idx].angle += particles[idx].spin*dt;

		// коллизия со стенами
			if (particles[idx].x < -offset+e)
				particles[idx].x = -offset+e;
			if (particles[idx].x > offset-e)
				particles[idx].x = offset-e;

			if (particles[idx].y < -offset+e)
				particles[idx].y = -offset+e;
			if (particles[idx].y > offset-e)
				particles[idx].y = offset-e;

			if (particles[idx].z < e)
				particles[idx].z = e;
			if (particles[idx].z > 2*offset-e)
				particles[idx].z = 2*offset-e;

		idx += blockDim.x * gridDim.x;
	}
}

__global__ void floor_kernel(uchar4* dev_floor_data, uint floor_texture_size, particle* particles, uint num_particles, particle projectile, float offset){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	float x, y;
	for (uint i = idx; i < floor_texture_size; i += blockDim.x*gridDim.x){
		for (uint j = idy; j < floor_texture_size; j += blockDim.y*gridDim.y){
			x = (2.0*i/(floor_texture_size-1.0)-1.0)*offset;
			y = (2.0*j/(floor_texture_size-1.0)-1.0)*offset;

			float fg = 0.0;
			for (uint k = 0; k < num_particles; ++k){
				fg += 100.0*particles[k].q/(sqr(x-particles[k].x)+sqr(y-particles[k].y)+sqr(particles[k].z));
			}
			fg += 100.0*projectile.q/(sqr(x-projectile.x)+sqr(y-projectile.y)+sqr(projectile.z));

			fg = min(max(0.0f, fg), 255.0f);
			dev_floor_data[j*floor_texture_size+i] = make_uchar4(0, (uint)fg, (uint)fg, 255);
		}
	}
}

void particles_step(){
	CSC(cudaMemcpy(dev_particles, particles, sizeof(particle)*num_particles, cudaMemcpyHostToDevice));
	particle_kernel<<<16, 32>>> (dev_particles, num_particles, cam_particle, projectile, K, W, g, offset, dt, e);
	CSC(cudaGetLastError());
	CSC(cudaMemcpy(particles, dev_particles, sizeof(particle)*num_particles, cudaMemcpyDeviceToHost));
}

void floor_step(){
	uchar4* dev_floor_data;
	size_t size;
	CSC(cudaGraphicsMapResources(1, &pbo_res, 0));		// Делаем буфер доступным для CUDA
	CSC(cudaGraphicsResourceGetMappedPointer((void**) &dev_floor_data, &size, pbo_res));	// Получаем указатель на память буфера
	floor_kernel<<<dim3(16, 16), dim3(32, 32)>>>(dev_floor_data, floor_texture_size, dev_particles, num_particles, projectile, offset);
	CSC(cudaGetLastError());
	CSC(cudaGraphicsUnmapResources(1, &pbo_res, 0));		// Возращаем буфер OpenGL'ю что бы он мог его использовать

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

	draw_projectile();
	draw_particles();
	draw_floor();
	
	glutSwapBuffers();
}

void update(){
	// camera
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

	// camera
		cam_particle.x = cam_x;
		cam_particle.y = cam_y;
		cam_particle.z = cam_z;
	// projectile
		projectile.x += projectile.dx*dt;
		projectile.y += projectile.dy*dt;
		projectile.z += projectile.dz*dt;
	// particles
		particles_step();

	// floor
		floor_step();

	glutPostRedisplay();
}

int main(int argc, char **argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(w, h);
	glutCreateWindow("Lab6");

	glutIdleFunc(update);
	glutDisplayFunc(display);
	glutKeyboardFunc(keys);
	glutPassiveMotionFunc(mouse);
	glutMouseFunc(mouse_click);
	glutReshapeFunc(reshape);

	glutSetCursor(GLUT_CURSOR_NONE);	// Скрываем курсор мышки

	// load particle texture
		int texture_w, texture_h;
		FILE *fp = fopen("mars.data", "rb");
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

	// setup floor texture
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
		glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeof(uchar4)*floor_texture_size*floor_texture_size, NULL, GL_DYNAMIC_DRAW);	// Задаем размер буфера
		cudaGraphicsGLRegisterBuffer(&pbo_res, pbo, cudaGraphicsMapFlagsWriteDiscard);				// Регистрируем буфер для использования его памяти в CUDA
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);			// Деактивируем буфер

	// init scene
		particles = (particle*)malloc(sizeof(particle)*num_particles);
		CSC(cudaMalloc(&dev_particles, sizeof(particle)*num_particles));
		particles_random_init();

	glutMainLoop();
}