아래는 각 SQL문을 한 줄(엔터) 단위로 분석한 내용입니다.

---

### create table sangdata(
**상품 데이터(sangdata) 테이블 생성 시작**

---

### code int primary key,
- `code`: 정수형, 기본키(Primary Key) 역할을 하는 상품 코드

### sang varchar(20),
- `sang`: 상품명, 최대 20자 문자열

### su int,
- `su`: 수량, 정수형

### dan int);                    참고 : 한글이 깨질 경우 ... dan int)charset=utf8;
- `dan`: 단가, 정수형  
- 테이블 생성 종료  
- 참고: 한글 깨짐 방지를 위해 charset=utf8 옵션을 사용할 수 있음

---

### insert into sangdata values(1,'장갑',3,10000);
- sangdata 테이블에 (1, '장갑', 3, 10000) 데이터 삽입

### insert into sangdata values(2,'벙어리장갑',2,12000);
- sangdata 테이블에 (2, '벙어리장갑', 2, 12000) 데이터 삽입

### insert into sangdata values(3,'가죽장갑',10,50000);
- sangdata 테이블에 (3, '가죽장갑', 10, 50000) 데이터 삽입

### insert into sangdata values(4,'가죽점퍼',5,650000);
- sangdata 테이블에 (4, '가죽점퍼', 5, 650000) 데이터 삽입

---

### create table buser(
**부서 데이터(buser) 테이블 생성 시작**

---

### buserno int primary key, 
- `buserno`: 부서 번호, 정수형, 기본키

### busername varchar(10) not null,
- `busername`: 부서명, 최대 10자, null 불가

### buserloc varchar(10),
- `buserloc`: 부서 위치, 최대 10자

### busertel varchar(15));
- `busertel`: 부서 전화번호, 최대 15자  
- 테이블 생성 종료

---

### insert into buser values(10,'총무부','서울','02-100-1111');
- buser 테이블에 (10, '총무부', '서울', '02-100-1111') 데이터 삽입

### insert into buser values(20,'영업부','서울','02-100-2222');
- buser 테이블에 (20, '영업부', '서울', '02-100-2222') 데이터 삽입

### insert into buser values(30,'전산부','서울','02-100-3333');
- buser 테이블에 (30, '전산부', '서울', '02-100-3333') 데이터 삽입

### insert into buser values(40,'관리부','인천','032-200-4444');
- buser 테이블에 (40, '관리부', '인천', '032-200-4444') 데이터 삽입

---

### create table jikwon(
**직원 데이터(jikwon) 테이블 생성 시작**

---

### jikwonno int primary key,
- `jikwonno`: 직원 번호, 정수형, 기본키

### jikwonname varchar(10) not null,
- `jikwonname`: 직원 이름, 최대 10자, null 불가

### busernum int not null,
- `busernum`: 소속 부서 번호, 정수형, null 불가

### jikwonjik varchar(10) default '사원', 
- `jikwonjik`: 직급, 최대 10자, 기본값 '사원'

### jikwonpay int,
- `jikwonpay`: 급여, 정수형

### jikwonibsail date,
- `jikwonibsail`: 입사일, 날짜형

### jikwongen varchar(4),
- `jikwongen`: 성별, 최대 4자

### jikwonrating char(3),
- `jikwonrating`: 등급, 3자

### CONSTRAINT ck_jikwongen check(jikwongen='남' or jikwongen='여'));
- 성별(jikwongen)은 '남' 또는 '여'만 허용하는 제약조건

---

### insert into jikwon values(1,'홍길동',10,'이사',9900,'2008-09-01','남','a');
- jikwon 테이블에 (1, '홍길동', 10, '이사', 9900, '2008-09-01', '남', 'a') 데이터 삽입

### ... (중간 생략, 동일 패턴)

---

### create table gogek(
**고객 데이터(gogek) 테이블 생성 시작**

---

### gogekno int primary key,
- `gogekno`: 고객 번호, 정수형, 기본키

### gogekname varchar(10) not null,
- `gogekname`: 고객 이름, 최대 10자, null 불가

### gogektel varchar(20),
- `gogektel`: 고객 전화번호, 최대 20자

### gogekjumin char(14),
- `gogekjumin`: 주민등록번호, 14자

### gogekdamsano int,
- `gogekdamsano`: 담당 직원 번호, 정수형

### CONSTRAINT FK_gogekdamsano foreign key(gogekdamsano) references jikwon(jikwonno));
- gogekdamsano는 jikwon 테이블의 jikwonno를 참조하는 외래키

---

### insert into gogek values(1,'이나라','02-535-2580','850612-1156777',5);
- gogek 테이블에 (1, '이나라', '02-535-2580', '850612-1156777', 5) 데이터 삽입

### ... (중간 생략, 동일 패턴)

---

### create table board(
**게시판 데이터(board) 테이블 생성 시작**

---

### num int primary key,
- `num`: 글 번호, 정수형, 기본키

### author varchar(10),
- `author`: 작성자, 최대 10자

### title varchar(50),
- `title`: 제목, 최대 50자

### content varchar(4000),
- `content`: 내용, 최대 4000자

### bwrite date,
- `bwrite`: 작성일, 날짜형

### readcnt int default 0);
- `readcnt`: 조회수, 정수형, 기본값 0

---

### insert into board(num,author,title,content,bwrite) values(1,'홍길동','연습','연습내용',now());
- board 테이블에 (1, '홍길동', '연습', '연습내용', 현재날짜) 데이터