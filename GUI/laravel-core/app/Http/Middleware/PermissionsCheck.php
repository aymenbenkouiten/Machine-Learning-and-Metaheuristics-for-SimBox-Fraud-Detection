<?php

namespace App\Http\Middleware;

use Closure;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Symfony\Component\HttpFoundation\Response;

class PermissionsCheck
{
    /**
     * Handle an incoming request.
     *
     * @param  \Closure(\Illuminate\Http\Request): (\Symfony\Component\HttpFoundation\Response)  $next
     */
    public function handle(Request $request, Closure $next, $permissions): Response
    {
        $user = Auth::user();
        if($user->permissions == null)
        {
            return response()->view('webmaster.invalidated');
        }
        else
        {
            if($user->Has_Permission($permissions))
            {
                return $next($request);
            }
            else
            {
                return abort(401);
            }
        }
    }
}
