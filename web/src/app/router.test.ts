import { describe, expect, it } from "vitest";
import { routeHref, routeOf, routeVehicle } from "./router";

describe("роутер", () => {
  it("экраны", () => {
    expect(routeOf("/")).toBe("search");
    expect(routeOf("/gallery")).toBe("gallery");
    expect(routeOf("/route/959")).toBe("route");
    expect(routeOf("/unknown")).toBe("search");
  });
  it("vehicle_id в адресе маршрута", () => {
    expect(routeHref("959")).toBe("/route/959");
    expect(routeVehicle("/route/959")).toBe("959");
    expect(routeVehicle(routeHref("a b"))).toBe("a b");
    expect(routeVehicle("/route/")).toBeNull();
    expect(routeVehicle("/route/%E0")).toBeNull();
  });
});
