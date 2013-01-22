#include "include.cuh"

bool updateDraw = 1;
bool showSphere = 1;

OpenGLCamera oglcamera(camreal3(0,0,-1),camreal3(0,0,0),camreal3(0,1,0),1);

//RENDERING STUFF
void changeSize(int w, int h) {
	if(h == 0) {h = 1;}
	float ratio = 1.0* w / h;
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glViewport(0, 0, w, h);
	gluPerspective(45,ratio,.1,1000);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0.0,0.0,0.0,		0.0,0.0,-7,		0.0f,1.0f,0.0f);
}

void initScene(){
	GLfloat light_position[] = { 1.0, 1.0, 1.0, 0.0 };
	glClearColor (1.0, 1.0, 1.0, 0.0);
	glShadeModel (GL_SMOOTH);
	glEnable(GL_COLOR_MATERIAL);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable (GL_POINT_SMOOTH);
	glEnable (GL_BLEND);
	glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glHint (GL_POINT_SMOOTH_HINT, GL_DONT_CARE);
}

void drawAll()
{
//	if(updateDraw){
//		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//		glEnable(GL_DEPTH_TEST);
//		glFrontFace(GL_CCW);
//		glCullFace(GL_BACK);
//		glEnable(GL_CULL_FACE);
//		glDepthFunc(GL_LEQUAL);
//		glClearDepth(1.0);
//
//		glPointSize(2);
//		glLoadIdentity();
//
//		oglcamera.Update();

//		glColor3f(0.0f,1.0f,0.0f);
//		glBegin(GL_QUADS);
//		double clip =10;
//		glVertex3f(clip,sys.groundHeight,clip);
//		glVertex3f(clip,sys.groundHeight,-clip);
//		glVertex3f(-clip,sys.groundHeight,-clip);
//		glVertex3f(-clip,sys.groundHeight,clip);
//		glEnd();
//		glFlush();

//		glColor3f(0.0f,0.0f,1.0f);
//		glPushMatrix();
//		float3 position = sys.getXYZPosition(100,0);
//		//cout << position.x << " " << position.y << " " << position.z << endl;
//		glTranslatef(position.x,position.y,position.z);
//		glutSolidSphere(1,10,10);
//		glPopMatrix();

//		for(int i=0;i<sys.elements.size();i++)
//		{
//			int xiDiv = sys.numContactPoints;
//
//			double xiInc = 1/(static_cast<double>(xiDiv-1));
//
//			if(!showSphere)
//			{
//				glColor3f(0.0f,0.0f,1.0f);
//				for(int j=0;j<xiDiv;j++)
//				{
//					glPushMatrix();
//					float3 position = sys.getXYZPosition(i,xiInc*j);
//					glTranslatef(position.x,position.y,position.z);
//					glutSolidSphere(sys.elements[i].getRadius(),10,10);
//					glPopMatrix();
//				}
//			}
//			else
//			{
//				int xiDiv = sys.numContactPoints;
//				double xiInc = 1/(static_cast<double>(xiDiv-1));
//				glLineWidth(sys.elements[i].getRadius()*500);
//				glColor3f(0.0f,1.0f,0.0f);
//				glBegin(GL_LINE_STRIP);
//				for(int j=0;j<sys.numContactPoints;j++)
//				{
//					float3 position = sys.getXYZPosition(i,xiInc*j);
//					glVertex3f(position.x,position.y,position.z);
//				}
//				glEnd();
//				glFlush();
//			}
//		}
//
//		glutSwapBuffers();
//	}
}

void renderSceneAll(){
	if(OGL){drawAll();}
}

void CallBackKeyboardFunc(unsigned char key, int x, int y) {
	switch (key) {
	case 'w':
		oglcamera.Forward();
		break;
	case 's':
		oglcamera.Back();
		break;

	case 'd':
		oglcamera.Right();
		break;

	case 'a':
		oglcamera.Left();
		break;

	case 'q':
		oglcamera.Up();
		break;

	case 'e':
		oglcamera.Down();
		break;
	}
}

void CallBackMouseFunc(int button, int state, int x, int y) {
	oglcamera.SetPos(button, state, x, y);
}
void CallBackMotionFunc(int x, int y) {
	oglcamera.Move2D(x, y);
}

int main(int argc, char** argv)
{
	custom_vector<realV> aabb_data;
	realV point = F3(0,0,0);
	aabb_data.push_back(point);
	point = F3(0,0,0);
	//aabb_data.push_back(point);
	point = F3(1,1,1);
	aabb_data.push_back(point);
	point = F3(1,1,1);
	//aabb_data.push_back(point);

	CollisionDetector* collisionManager = new CollisionDetector(aabb_data);
	collisionManager->detectPossibleCollisions();
//	while(true)
//	{
//		collisionManagerPtr->detectPossibleCollisions();
//	}

//#pragma omp parallel sections
//	{
//#pragma omp section
//		{
////			while(true)
////			{
////				collisionManagerPtr->detectPossibleCollisions();
////			}
//		}
//#pragma omp section
//		{
//			if(OGL){
//				glutInit(&argc, argv);
//				glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
//				glutInitWindowPosition(0,0);
//				glutInitWindowSize(1024	,512);
//				glutCreateWindow("SSCD");
//				glutDisplayFunc(renderSceneAll);
//				glutIdleFunc(renderSceneAll);
//				glutReshapeFunc(changeSize);
//				glutIgnoreKeyRepeat(0);
//				glutKeyboardFunc(CallBackKeyboardFunc);
//				glutMouseFunc(CallBackMouseFunc);
//				glutMotionFunc(CallBackMotionFunc);
//				initScene();
//				glutMainLoop();
//			}
//		}
//	}

	return 0;
}
