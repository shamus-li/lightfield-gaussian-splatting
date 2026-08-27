const BASE_PATH = "/lightfield-gaussian-splatting";

export default {
  async fetch(request, env) {
    const requestUrl = new URL(request.url);

    if (requestUrl.pathname === BASE_PATH) {
      requestUrl.pathname = `${BASE_PATH}/`;
      return Response.redirect(requestUrl.toString(), 308);
    }

    if (!requestUrl.pathname.startsWith(`${BASE_PATH}/`)) {
      return fetch(request);
    }

    const assetUrl = new URL(request.url);
    assetUrl.pathname = requestUrl.pathname.slice(BASE_PATH.length);
    const response = await env.ASSETS.fetch(new Request(assetUrl, request));
    const location = response.headers.get("Location");

    if (!location) {
      return response;
    }

    const redirectUrl = new URL(location, requestUrl.origin);
    if (
      redirectUrl.origin !== requestUrl.origin ||
      redirectUrl.pathname.startsWith(`${BASE_PATH}/`)
    ) {
      return response;
    }

    redirectUrl.pathname = `${BASE_PATH}${redirectUrl.pathname}`;
    const headers = new Headers(response.headers);
    headers.set("Location", redirectUrl.toString());
    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers,
    });
  },
};
