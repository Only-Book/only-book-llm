import os
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Book
from .serializers import BookSerializer
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

VECTOR_DB_PATH = os.path.join(os.getcwd(), 'vectorDB')
OPENAI_API_KEY = settings.OPENAI_API_KEY

@api_view(['GET'])
def get_books(request):
    books = Book.objects.all()
    serializer = BookSerializer(books, many=True)
    return Response(serializer.data)

def embed_books():
    # 데이터베이스에서 모든 책 정보 가져오기
    books = Book.objects.all()
    documents = []

    # 각 책에 대해 Document 객체 생성
    for book in books:
        page_content = f"Title: {book.title}, Author: {book.author}, Description: {book.description}"
        documents.append(Document(page_content=page_content, metadata={"id": str(book.id)}))

    # OpenAI를 사용하여 임베딩 생성
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    # Chroma 벡터 스토어 초기화 및 업데이트
    chroma = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
    chroma.add_documents(documents)
    print(f"벡터 스토어에 임베딩된 책의 수: {len(documents)}")


@api_view(['POST'])
def chatbot_response(request):
    user_input = request.data.get('message', '')
    print("유저의 질문: " + user_input)

    # 벡터 스토어 초기화 (이미 임베딩된 데이터를 사용)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    chroma = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)

    # 사용자의 입력을 임베딩하고 벡터 스토어에서 최대 5개의 결과 검색
    query_embedding = embeddings.embed_query(user_input)
    search_results = chroma.similarity_search_by_vector(query_embedding, k=5)

    # 유사도 검색 결과 출력
    print("유사도 검색 결과:")
    for result in search_results:
        print(f"ID: {result.metadata['id']}, Content: {result.page_content}")

    # 검색 결과가 없을 경우
    if not search_results:
        return Response({"response": "적합한 책을 찾지 못했습니다. 죄송합니다."})

    # 데이터베이스에서 책 정보 가져오기, 중복된 ID 제거
    book_ids = list(set(int(result.metadata["id"]) for result in search_results))  # ID를 정수형으로 변환
    print(f"검색된 책 ID 목록: {book_ids}")

    # 책 조회 시 ID를 문자열로 변환하여 쿼리 시도
    recommended_books = Book.objects.filter(id__in=book_ids[:3])  
    print(f"조회된 책: {recommended_books}")

    # 책이 조회되지 않은 경우 처리
    if not recommended_books.exists():
        return Response({"response": "적합한 책을 찾지 못했습니다. 죄송합니다."})

    serializer = BookSerializer(recommended_books, many=True)

    # 책 정보 바탕으로 LLM에 질문 생성
    llm = ChatOpenAI(api_key=OPENAI_API_KEY)
    books_info = [
        f"Title: {book['title']}, Author: {book['author']}, Description: {book['description']}"
        for book in serializer.data
    ]
    books_info_text = ' | '.join(books_info)
    print("검색된 책: " + books_info_text)
    question = (
        f"사용자가 '{user_input}'라고 말했습니다. 사용자의 관심사에 맞는 책을 추천해드리기 위해 다음 도서를 선정했습니다: {books_info_text}. "
        f"답변을 해줄 때 사용자의 질문에 공감하고 최대한 정성스럽게 답변해주세요. 이모지도 마음껏 사용해주세요."
        f"사용자의 마음을 이해하고 공감하며 아주 친절하고 공감하는 투로 답변해주세요."
        f"추천한 책 중에서 특히 어떤 점이 이 책들을 돋보이게 하는지, 책의 주요 내용과 독자에게 어떤 유익을 줄 수 있는지 상세히 설명해주세요. "
        f"사용자의 질문을 절대로 바꿔선 안됩니다. 사용자의 질문을 기반으로 답변을 해야합니다."
        f"시작은 꼭 '{user_input}'라는 질문을 주셨군요! 로 시작하세요."
    )

    # LLM에 질문하고 응답 받기
    llm_response = llm([HumanMessage(content=question)])
    response_message = llm_response.content

    return Response({"response": response_message, "books": serializer.data})
