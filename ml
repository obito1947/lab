Ex.No-1
	Obtain The Topological Ordering of Vertices in a Given Digraph
	Date:

Source Code:

#include<stdio.h>
int temp[10],k=0;
voidtopo(intn,intindegree[10],int a[10][10])
{
inti,j;
for(i=1;i<=n;i++)
{
if(indegree[i]==0)
{
indegree[i]=1;
temp[++k]=i;
for(j=1;j<=n;j++)
{
if(a[i][j]==1&&indegree[j]!=-1)
indegree[j]--;
}
i=0;
}
}
}
void main()
{
inti,j,n,indegree[10],a[10][10];
printf("enter the number of vertices:");
scanf("%d",&n);
for(i=1;i<=n;i++)
indegree[i]=0;
printf("\n enter the adjacency matrix\n");
for(i=1;i<=n;i++)
for(j=1;j<=n;j++)
{
scanf("%d",&a[i][j]);
if(a[i][j]==1)
indegree[j]++;
}
topo(n,indegree,a);
if(k!=n)
printf("topological ordering is not possible\n");
else
{
printf("\n topological ordering is :\n");
for(i=1;i<=k;i++)
printf("v%d\t",temp[i]);
}
}

Using DFS algorithm:

1.	Run DFS(G), computing finish time for each vertex
2.	As each vertex is finished, insert it onto the front of a list

Source Code:

#include<stdio.h>
inti,visit[20],n,adj[20][20],s,topo_order[10];
voiddfs(int v)
{
int w;
visit[v]=1;
for(w=1;w<=n;w++)
if((adj[v][w]==1) && (visit[w]==0))
dfs(w);
topo_order[i--]=v;
}
voidmain ()
{
intv,w;
printf("Enter the number of vertices:\n");
scanf("%d",&n);
printf("Enter the adjacency matrix:\n");
for(v=1;v<=n;v++)
for(w=1;w<=n;w++)
scanf("%d",&adj[v][w]);
for(v=1;v<=n;v++)
visit[v]=0;
i=n;
for(v=1;v<=n;v++)
{
if(visit[v]==0)
dfs(v);
}
printf("\nTopological sorting is:");
for(v=1;v<=n;v++)
printf("v%d ",topo_order[v]);
}
  
\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

Ex.No-2
	Sort a given set of elements using the Quick sort method and determine the time required to sort the elements.	Date:

Source code:
#include<stdio.h>
#include<conio.h>
void quicksort(int[],int, int);
intpartition (int[],int,int);
void main()
{
inti,n,a[20],ch=1;
clrscr();
while(ch)
{
printf("\n enter the number of elements\n");
scanf("%d",&n);
printf("\n enter the array elements\n");
for(i=0;i<n;i++)
scanf("%d",&a[i]);
quicksort(a,0,n-1);
printf("\n\nthe sorted array elements are\n\n");
for(i=0;i<n;i++)
printf("\n%d",a[i]);
printf("\n\n do u wish to continue (0/1)\n");
scanf("%d",&ch);
}
getch();
}
void quicksort(int a[],intlow,int high)
{
int mid;
if(low<high)
{
mid=partition(a,low,high);
quicksort(a,low,mid-1);
quicksort(a,mid+1,high);
}
}
int partition(int a[],intlow,int high)
{
intkey,i,j,temp,k;
key=a[low];
i=low+1;
j=high;
while(i<=j)
{
while(i<=high && key>=a[i])
i=i+1;
while(key<a[j])
j=j-1;
if(i<j)    
{
temp=a[i];
a[i]=a[j];
a[j]=temp;
}
else
{
k=a[j];
a[j]=a[low];
a[low]=k;
}
}
return j;
}
////////////////////////////////////////////////////////////////////////////////////////
Ex.No-3
	Sort a given set of elements using the Merge sort method and determine the time required to sort the elements.	Date:

Source Code:

#include<stdio.h>
#include<conio.h>
#include<time.h>
#define max 20
voidmergesort(int a[],intlow,int high);
void merge(int a[],intlow,intmid,int high);
void main()
{
intn,i,a[max],ch=1;
clock_tstart,end;
clrscr();
while(ch)
{
printf("\n\t enter the number of elements\n");
scanf("%d",&n);
printf("\n\t enter the elements\n");
for(i=0;i<n;i++)
scanf("%d",&a[i]);
start= clock();
mergesort(a,0,n-1);
end=clock();
printf("\nthe sorted array is\n");
for(i=0;i<n;i++)
printf("%d\n",a[i]);
printf("\n\ntime taken=%lf",(end-start)/CLK_TCK);
printf("\n\ndo u wish to continue(0/1) \n");
scanf("%d",&ch);
}
getch();
}  voidmergesort(int a[],intlow,int high) {
int mid;
delay(100);
if(low<high) {
mid=(low+high)/2;
mergesort(a,low,mid);
mergesort(a,mid+1,high);
merge(a,low,mid,high);
}
}
void merge(int a[],intlow,intmid,int high) {
inti,j,k,t[max];
i=low;
j=mid+1;
k=low;
while((i<=mid) && (j<=high))
if(a[i]<=a[j])
t[k++]=a[i++];
else
t[k++]=a[j++];
while(i<=mid)
t[k++]=a[i++];
while(j<=high)
t[k++]=a[j++];
for(i=low;i<=high;i++)
a[i]=t[i];
}
OUTPUT:
///////////////////////////////////////////////////////////////////////////////
Ex.No-4
	Check whether a given graph is connected or not using DFS method.	Date:

Source code:

#include<stdio.h>
int visit[20],n,adj[20][20],s,count=0;
voiddfs(int v)
{
int w;
visit[v]=1;
count++;
for(w=1;w<=n;w++)
if((adj[v][w]==1) && (visit[w]==0))
dfs(w);
}
void main()
{
intv,w;
printf("Enter the no.of vertices:");
scanf("%d",&n);
printf("Enter the adjacency matrix:\n");
for(v=1;v<=n;v++)
for(w=1;w<=n;w++)
scanf("%d",&adj[v][w]);
for(v=1;v<=n;v++)
visit[v]=0;
dfs(1);
if(count==n)
printf("\nThe graph is connected");
else
printf("The graph is not connected");
}
//////////////////////////////////////////////////////
Ex.No-5
	Print all the nodes reachable from a given starting node in a directed graph using BFS method	Date:

Source code:

#include<stdio.h>
#define size 20
#define true 1
#define false 0
int queue[size],visit[20],rear=-1,front=0;
intn,s,adj[20][20],flag=0;
voidinsertq(int v)
{
queue[++rear]=v;
}
intdeleteq()
{
return(queue[front++]);
}
intqempty()
{
if(rear<front)
return 1;
else
return 0;
}
voidbfs(int v)
{
int w;
visit[v]=1;
insertq(v);
while(!qempty())
{
v=deleteq();
for(w=1;w<=n;w++)

if((adj[v][w]==1) && (visit[w]==0))
{
visit[w]=1;
flag=1;
printf("v%d\t",w);
insertq(w);
}
}
}
void main()
{
intv,w;
printf("Enter the no.of vertices:\n");
scanf("%d",&n);
printf("Enter adjacency matrix:");
for(v=1;v<=n;v++)
{
for(w=1;w<=n;w++)
scanf("%d",&adj[v][w]);
}
printf("Enter the start vertex:");
scanf("%d",&s);
printf("Reachability of vertex %d\n",s);
for(v=1;v<=n;v++)
visit[v]=0;
bfs(s);
if(flag==0)
{
printf("No path found!!\n");
}
}

OUTPUT:
//////////////////////////////////////////////////////////////////////////////////////////////////
Ex.No-6
	Find Minimum Cost Spanning Tree of a given undirected graph using Prim’s algorithm.	Date:

Source code:

# include <stdio.h>
int Prim (int g[20][20], int n, int t[20][20])
{
intu,v, min, mincost;
int visited[20];
inti,j,k;
visited[1] = 1;
for(k=2; k<=n; k++)
visited[k] = 0 ;
mincost = 0;
for(k=1; k<=n-1; k++)
{
min= 99;
u=1;
v=1;
for(i=1; i<=n; i++)
if(visited[i]==1)
for(j=1; j<=n; j++)
if( g[i][j] < min )
{
min = g[i][j];
u = i;    v = j;
}
t[u][v] = t[v][u] = g[u][v] ;
mincost = mincost + g[u][v] ;
visited[v] = 1;
printf("\n (%d, %d) = %d", u, v, t[u][v]);
for(i=1; i<=n; i++)
for(j=1; j<=n; j++)
if( visited[i] && visited[j] )
g[i][j] = g[j][i] = 99;
}
return(mincost);
}
void main()
{
int n, cost[20][20], t[20][20];
intmincost,i,j;
printf("\nEnter the no of nodes: ");
scanf("%d",&n);
printf("Enter the cost matrix:\n");
for(i=1; i<=n; i++)
for(j=1;j<=n;j++)
{
scanf("%d",&cost[i][j]);
if(cost[i][j]==0)
cost[i][j]=99;
}
for(i=1; i<=n; i++)
for(j=1; j<=n; j++)
t[i][j] = 99;
printf("\nThe order of Insertion of edges:");
mincost = Prim (cost,n,t);
printf("\nMinimum cost = %d\n\n", mincost);
}

OUTPUT:



/////////////////////////////////////////////////////////////
Ex.No-7
	Find Minimum Cost Spanning Tree of a given undirected graph using Kruskal's algorithm.	Date:

Source Code: 
#include<stdio.h>
int parent[20],min,mincost=0,ne=1,n,cost[20][20];
inta,b,i,j,u,v;
void main()
{
printf("Enter the no of nodes\n");
scanf("%d",&n);
printf("Enter the cost matrix\n");
for(i=1;i<=n;i++)
for(j=1;j<=n;j++)
{
scanf("%d",&cost[i][j]);
if(cost[i][j]==0)
cost[i][j]=99;
}
while(ne <n)
{
for(i=1,min=99;i<=n;i++)
for(j=1;j<=n;j++)
if(cost[i][j] < min)
{
min=cost[i][j];
a=u=i;
b=v=j;
}
While (parent[u])
u=parent[u];
while(parent[v])
v=parent[v];
if(u!=v)
{
printf("%d\t edge \t (%d,%d)=%d\n",ne++,a,b,min);
mincost+=min;
parent[v]=u;
}
cost[a][b]=cost[b][a]=99;
}
printf("The minimum cost=%d\n",mincost);
}

OUTPUT:




OUTPUT:


