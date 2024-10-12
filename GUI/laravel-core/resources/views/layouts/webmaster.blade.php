@extends('layouts.base')
@section('head')
<title>@yield('subtitle') - {{config('webmaster.title')}}</title>
<link href="{{asset('assets/css/app.css')}}" rel="stylesheet">
<link href="{{asset('assets/css/custom.css')}}" rel="stylesheet">
@endsection
@section('body')
<div class="wrapper">
    @include('components.sidebar')
    <div class="main">
        @include('components.navbar')
        <main class="content">
            @yield('content')
        </main>
        @include('components.footer')
    </div>
</div>
<script src="{{asset('assets/js/app.js')}}"></script>
@yield('script')

@if(session('success'))
    <script>
        document.addEventListener("DOMContentLoaded", function() {
                var message = '{{ session('success') }}';
                var type = "success";
                var duration = 5000;
                var ripple = false;
                var dismissible = true;
                var positionX = "right";
                var positionY = "top";
                window.notyf.open({type, message, duration, ripple, dismissible, position:{x: positionX, y: positionY}});
        });
        document.documentElement.style.setProperty('--main-color', '#{{config("settings.webmasterColor")}}');
    </script>
@endif
@if(session('error'))
    <script>
        document.addEventListener("DOMContentLoaded", function() {
                var message = '{{ session('error') }}';
                var type = "danger";
                var duration = 5000;
                var ripple = false;
                var dismissible = true;
                var positionX = "right";
                var positionY = "top";
                window.notyf.open({type, message, duration, ripple, dismissible, position:{x: positionX, y: positionY}});
        });
        document.documentElement.style.setProperty('--main-color', '#{{config("settings.webmasterColor")}}');
    </script>
@endif

@endsection